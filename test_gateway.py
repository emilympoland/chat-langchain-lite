import importlib
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import mock_open, patch

import httpx
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from scripts import setup


DIAGNOSTIC = (
    "LangSmith gateway could not reach Anthropic: add workspace secret "
    "ANTHROPIC_API_KEY in LangSmith Settings > Secrets"
)


def provider_error(status, message):
    response = httpx.Response(
        status, request=httpx.Request("POST", "https://gateway.example/messages")
    )
    return APIStatusError(message, response=response, body={"error": message})


class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (
            patch.dict(os.environ, LANGSMITH_API_KEY_GATEWAY="test-key"),
            patch("context.get_prompt", return_value="Test prompt"),
            patch("langchain.chat_models.init_chat_model"),
        ):
            cls.agent = importlib.import_module("agent.agent")
            cls.models = importlib.import_module("utils.models")

    def run_setup(self, error=None):
        output = io.StringIO()
        with (
            patch.dict(os.environ, LANGSMITH_API_KEY="test-key"),
            patch("sys.argv", ["setup", "--skip-baseline-experiments"]),
            patch.object(self.models.model, "invoke", side_effect=error) as ping,
            patch("utils.context_hub.push_agents_md"),
            patch("utils.context_hub.push_demo_skills"),
            patch.object(setup, "ensure_project_exists") as project,
            patch.object(setup, "setup_dataset"),
            patch.object(setup, "setup_online_evaluators", return_value=[]) as evaluators,
            patch("builtins.open", mock_open()),
            redirect_stdout(output),
        ):
            if error:
                with self.assertRaises(SystemExit) as caught:
                    setup.main()
                self.assertNotEqual(caught.exception.code, 0)
                project.assert_not_called()
                evaluators.assert_not_called()
            else:
                setup.main()
                project.assert_called_once()
                evaluators.assert_called_once()
            ping.assert_called_once_with("ping", max_tokens=1)
        return output.getvalue()

    def test_successful_preflight_allows_setup(self):
        self.assertIn("Setup complete", self.run_setup())

    def test_missing_workspace_secret_stops_setup(self):
        error = provider_error(424, "Secret ANTHROPIC_API_KEY not found in workspace")
        self.assertIn(DIAGNOSTIC, self.run_setup(error))

    def test_other_preflight_failures_retain_cause(self):
        for error in (
            provider_error(401, "Invalid gateway credentials"),
            provider_error(424, "Workspace secret OPENAI_API_KEY missing"),
            provider_error(424, "Workspace secret ANTHROPIC_API_KEY is invalid"),
            RuntimeError("connection unavailable"),
        ):
            with self.subTest(error=error):
                output = self.run_setup(error)
                self.assertIn(str(error), output)
                self.assertNotIn(DIAGNOSTIC, output)

    def test_model_configuration_failure_is_reported(self):
        with patch.dict("sys.modules", {"utils.models": None}), redirect_stdout(
            io.StringIO()
        ) as output:
            with self.assertRaises(SystemExit) as caught:
                setup.check_gateway()
        self.assertNotEqual(caught.exception.code, 0)
        self.assertIn("gateway preflight failed", output.getvalue())
        self.assertNotIn(DIAGNOSTIC, output.getvalue())

    def test_agent_logs_and_preserves_provider_failure(self):
        for status, message, missing in (
            (424, "Workspace secret ANTHROPIC_API_KEY is missing", True),
            (429, "Rate limit exceeded", False),
            (424, "Workspace secret OPENAI_API_KEY is missing", False),
        ):
            with self.subTest(status=status, message=message):
                error = provider_error(status, message)
                with (
                    patch.object(self.agent, "build_agent") as build,
                    self.assertLogs(self.agent.__name__, level="ERROR") as logs,
                ):
                    build.return_value.invoke.side_effect = error
                    with self.assertRaises(APIStatusError) as caught:
                        self.agent.invoke_agent("hello")
                self.assertIs(caught.exception, error)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(caught.exception.body, {"error": message})
                self.assertIn(message, str(caught.exception))
                self.assertIn(DIAGNOSTIC if missing else message, "\n".join(logs.output))
                self.assertIn(
                    DIAGNOSTIC if missing else message,
                    "\n".join(caught.exception.__notes__),
                )
                if not missing:
                    self.assertNotIn(DIAGNOSTIC, "\n".join(logs.output))

    def test_successful_invocation_preserves_return_contract(self):
        messages = [
            HumanMessage(content="hello"),
            ToolMessage(content="tool result", tool_call_id="tool-1", name="lookup_concept"),
            AIMessage(content="answer"),
        ]
        with patch.object(self.agent, "build_agent") as build:
            build.return_value.invoke.return_value = {"messages": messages}
            result = self.agent.invoke_agent("hello", thread_id="thread-1")
        self.assertEqual(result, {
            "output": "answer", "tools_called": ["lookup_concept"], "messages": messages
        })


if __name__ == "__main__":
    unittest.main()
