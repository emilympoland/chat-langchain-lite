import importlib
import json
import os
import unittest
from unittest.mock import patch

import httpx
from anthropic import Anthropic
from langchain.agents.middleware import AgentMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.callbacks import BaseCallbackHandler


class RunMetadata(BaseCallbackHandler):
    def __init__(self):
        self.root = []
        self.llm = []

    def on_chain_start(
        self, serialized, inputs, *, parent_run_id=None, metadata=None, **kwargs
    ):
        if parent_run_id is None:
            self.root.append(metadata)

    def on_chat_model_start(self, serialized, messages, *, metadata=None, **kwargs):
        self.llm.append(metadata)


class AgentModelSelectionTests(unittest.TestCase):
    def test_invocation_and_streaming_use_selected_model(self):
        with patch("context.get_prompt", return_value="Answer briefly."):
            agent = importlib.import_module("agent.agent")
        requests = []
        recorder = RunMetadata()

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            message = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": payload["model"],
                "content": [{"type": "text", "text": "Hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
            if not payload.get("stream"):
                return httpx.Response(200, json=message)
            events = [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {**message, "content": [], "stop_reason": None},
                    },
                ),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "Hello"},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join(
                    f"event: {event}\ndata: {json.dumps(data)}\n\n"
                    for event, data in events
                ),
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:

            def make_model(**kwargs):
                model = init_chat_model(**kwargs, callbacks=[recorder])
                model._client = Anthropic(
                    api_key=kwargs["api_key"],
                    base_url=kwargs["base_url"],
                    http_client=client,
                )
                os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = "changed-during-build"
                return model

            original_config = agent._config

            def config(*args):
                return {**original_config(*args), "callbacks": [recorder]}

            with (
                patch.dict(
                    os.environ,
                    {
                        "LANGSMITH_API_KEY_GATEWAY": "test-key",
                        "LANGSMITH_TRACING": "false",
                    },
                ),
                patch("utils.models.init_chat_model", side_effect=make_model),
                patch.object(
                    agent, "_readonly_context_hub_fs", return_value=AgentMiddleware()
                ),
                patch.object(agent, "_config", side_effect=config),
                patch.object(agent, "SYSTEM_PROMPT", "Answer briefly."),
            ):
                for streaming in (False, True):
                    for override in (
                        "",
                        "claude-sonnet-4-6",
                        "claude-haiku-4-5-20251001",
                    ):
                        with self.subTest(streaming=streaming, override=override):
                            os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = override
                            expected = override or "claude-haiku-4-5-20251001"
                            if streaming:
                                self.assertEqual(
                                    "".join(agent.stream_agent("Hello", "thread-test")),
                                    "Hello",
                                )
                            else:
                                self.assertTrue(
                                    agent.invoke_agent("Hello", "thread-test")[
                                        "messages"
                                    ]
                                )
                            self.assertEqual(requests[-1]["model"], expected)
                            self.assertEqual(recorder.root[-1]["model"], expected)
                            self.assertEqual(
                                recorder.root[-1]["thread_id"], "thread-test"
                            )
                            self.assertEqual(
                                recorder.llm[-1]["ls_model_name"], expected
                            )


if __name__ == "__main__":
    unittest.main()
