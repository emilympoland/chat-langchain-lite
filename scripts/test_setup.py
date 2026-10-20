import unittest
from unittest.mock import Mock, patch

from langchain_core.prompts import ChatPromptTemplate

from scripts.setup import EVALUATORS, create_online_evaluator


PROJECT_ID = "2099b370-87ee-4962-b6fd-eeca0b7a1093"


class OnlineEvaluatorConfigTests(unittest.TestCase):
    def create_config(self, ev):
        response = Mock(status_code=201)
        response.json.return_value = {"id": "rule-id"}
        with patch("scripts.setup.requests.post", return_value=response) as post:
            rule_id = create_online_evaluator("test-key", ev, PROJECT_ID, {})
        self.assertEqual(rule_id, "rule-id")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["session_id"], PROJECT_ID)
        self.assertEqual(payload["sampling_rate"], 1.0)
        return payload["evaluators"][0]["structured"]

    def test_scope_evaluators_receive_original_request_and_response(self):
        cases = [
            (
                "Write a Django OAuth application.",
                "That is outside my LangChain scope. I can help with LangGraph setup instead.",
            ),
            (
                "How do I set up LangGraph?",
                "Install LangGraph and then configure the",
            ),
            (
                "What is LangGraph's minimum Python version?",
                "LangGraph requires Python 3.10+.",
            ),
            (
                "How do I set up LangSmith tracing?",
                "Set LANGSMITH_TRACING=true and configure your API key.",
            ),
        ]
        for ev in EVALUATORS:
            if ev["feedback_key"] not in {"response_completeness", "tool_usage"}:
                continue
            config = self.create_config(ev)
            for question, answer in cases:
                with self.subTest(evaluator=ev["feedback_key"], question=question):
                    run = {
                        "input": {"messages": [{"role": "user", "content": question}]},
                        "output": {"output": answer, "tools_called": []},
                    }
                    variables = {
                        name: run[source] for name, source in config["variable_mapping"].items()
                    }
                    prompt = ChatPromptTemplate.from_messages(
                        [tuple(message) for message in config["prompt"]], template_format="mustache"
                    )
                    message = prompt.format_messages(**variables)[-1].content
                    self.assertIn(question, message)
                    self.assertIn(answer, message)
                    self.assertIn("tools_called", message)

    def test_other_evaluators_remain_response_only_and_schemas_are_unchanged(self):
        for ev in EVALUATORS:
            with self.subTest(evaluator=ev["feedback_key"]):
                config = self.create_config(ev)
                self.assertEqual(config["schema"], {
                    "title": "score_run",
                    "type": "object",
                    "properties": {
                        ev["feedback_key"]: {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "1 = pass, 0 = fail",
                        }
                    },
                    "required": [ev["feedback_key"]],
                })
                if ev["feedback_key"] not in {"response_completeness", "tool_usage"}:
                    self.assertEqual(config["variable_mapping"], {"output": "output"})
                    self.assertEqual(config["prompt"], [
                        ["system", ev["system_prompt"]],
                        ["human", "Agent response: {{output}}"],
                    ])


if __name__ == "__main__":
    unittest.main()
