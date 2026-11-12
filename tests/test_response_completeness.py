import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from utils.streaming import TRUNCATION_NOTICE


class ResponseCompletenessTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        with (
            patch("context.get_prompt", return_value=""),
            patch.dict("os.environ", {"LANGSMITH_API_KEY_GATEWAY": "test-key"}),
        ):
            cls.agent = importlib.import_module("agent.agent")
        cls.web = importlib.import_module("web.app")

    def test_model_budget(self):
        from utils.models import model

        self.assertEqual(model.max_tokens, 2048)
        self.assertEqual(model.temperature, 0)

    def test_invocation(self):
        text = "A substantive explanation. " * 400
        for metadata in (
            {"stop_reason": "end_turn"},
            {"stop_reason": "max_tokens"},
            {"stop_reason": "tool_use"},
            {},
        ):
            with self.subTest(metadata=metadata):
                message = AIMessage(
                    content=[{"type": "text", "text": text}], response_metadata=metadata
                )
                with patch.object(self.agent, "build_agent") as build:
                    build.return_value.invoke.return_value = {
                        "messages": [HumanMessage("question"), message]
                    }
                    result = self.agent.invoke_agent("question")
                expected = text + (
                    TRUNCATION_NOTICE
                    if metadata.get("stop_reason") == "max_tokens"
                    else ""
                )
                self.assertEqual(result["output"], expected)

    def test_streaming(self):
        text = "A substantive explanation. " * 400
        for metadata in (
            {"stop_reason": "end_turn"},
            {"stop_reason": "max_tokens"},
            {"stop_reason": "tool_use"},
            {},
        ):
            with self.subTest(metadata=metadata):
                chunks = [
                    AIMessageChunk(
                        content="",
                        id="tool",
                        response_metadata={"stop_reason": "max_tokens"},
                    ),
                    AIMessageChunk(content=text, id="final"),
                    AIMessageChunk(content="", id="final", response_metadata=metadata),
                    AIMessageChunk(content="", id="final"),
                ]
                with patch.object(self.agent, "build_agent") as build:
                    build.return_value.stream.return_value = [
                        (chunk, {}) for chunk in chunks
                    ]
                    output = "".join(self.agent.stream_agent("question"))
                self.assertEqual(
                    output,
                    text
                    + (
                        TRUNCATION_NOTICE
                        if metadata.get("stop_reason") == "max_tokens"
                        else ""
                    ),
                )

    async def test_web_streaming(self):
        for metadata in (
            {"stop_reason": "end_turn"},
            {"stop_reason": "max_tokens"},
            {"stop_reason": "tool_use"},
            {},
            None,
        ):
            with self.subTest(metadata=metadata):

                async def join_stream(*args, **kwargs):
                    for msg in [
                        {
                            "id": "tool",
                            "content": "",
                            "response_metadata": {"stop_reason": "max_tokens"},
                        },
                        {
                            "id": "final",
                            "content": [{"type": "text", "text": "Answer"}],
                        },
                        {"id": "final", "content": "", "response_metadata": metadata},
                        {"id": "final", "content": ""},
                    ]:
                        yield SimpleNamespace(
                            event="messages",
                            data=[{"type": "AIMessageChunk", **msg}, {}],
                        )

                client = SimpleNamespace(runs=SimpleNamespace(join_stream=join_stream))
                with patch.object(self.web, "get_client", return_value=client):
                    response = await self.web.stream(
                        {"thread": "test-thread"},
                        run="12345678-1234-1234-1234-123456789abc",
                    )
                    frames = [frame async for frame in response.body_iterator]
                tokens = [frame for frame in frames if "event: token" in frame]
                expected = "Answer" + (
                    TRUNCATION_NOTICE
                    if (metadata or {}).get("stop_reason") == "max_tokens"
                    else ""
                )
                self.assertEqual(
                    tokens[-1], self.web.sse_message(expected, event="token")
                )
                self.assertEqual(
                    sum(
                        TRUNCATION_NOTICE.splitlines()[-1] in frame for frame in tokens
                    ),
                    int((metadata or {}).get("stop_reason") == "max_tokens"),
                )
                self.assertIn("event: done", frames[-1])


if __name__ == "__main__":
    unittest.main()
