import importlib
import os
import unittest
from unittest.mock import patch

from anthropic.types import Message
from langchain.agents.middleware import AgentMiddleware


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, LANGSMITH_API_KEY_GATEWAY="test-key"), patch(
            "dotenv.load_dotenv"
        ):
            cls.models = importlib.import_module("utils.models")
        with patch("context.get_prompt", return_value="Answer LangGraph questions."):
            cls.agent = importlib.import_module("agent.agent")

    def test_shared_model_limits(self):
        self.assertEqual(self.models.model.max_tokens, 4096)
        self.assertEqual(self.models.model.temperature, 0)

    def test_code_walkthrough_completes_through_agent(self):
        walkthrough = (
            "Define a typed state and add a node that returns state updates. " * 40
            + "\n```python\ngraph_builder.add_edge(START, 'chat')\n"
            "graph_builder.add_edge('chat', END)\n"
            "graph = graph_builder.compile(checkpointer=checkpointer)\n```\n"
            "Use middleware around model calls, persist state with a checkpointer, "
            "and consume graph.stream for streaming updates. Walkthrough complete."
        )
        self.assertGreater(len(walkthrough.split()), 300)

        def respond(**kwargs):
            self.assertEqual(kwargs["max_tokens"], 4096)
            self.assertEqual(kwargs["temperature"], 0)
            self.assertIn("LangGraph", kwargs["messages"][-1]["content"])
            return Message.model_validate({
                "id": "msg_walkthrough",
                "type": "message",
                "role": "assistant",
                "model": self.models.MODEL_CONFIG["model"],
                "content": [{"type": "text", "text": walkthrough}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 30, "output_tokens": 750},
            })

        with patch.dict(os.environ, LANGSMITH_TRACING="false"), patch.object(
            self.models.model._client.messages, "create", side_effect=respond
        ), patch.object(
            self.agent, "_readonly_context_hub_fs", return_value=AgentMiddleware()
        ):
            result = self.agent.invoke_agent(
                "Give a LangGraph code walkthrough including middleware, persistence, "
                "and streaming."
            )

        self.assertEqual(result["output"], walkthrough)
        final = result["messages"][-1]
        self.assertEqual(final.response_metadata["stop_reason"], "end_turn")
        self.assertGreater(final.usage_metadata["output_tokens"], 300)
        self.assertTrue(result["output"].endswith("Walkthrough complete."))


if __name__ == "__main__":
    unittest.main()
