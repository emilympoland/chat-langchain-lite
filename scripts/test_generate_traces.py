"""Tests for trace generator setup and error recovery."""

import contextlib
import io
import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

import anthropic
import httpx
from langchain_anthropic import ChatAnthropic
from langsmith import get_tracing_context

from scripts import generate_traces


def status_error(status, error_type=anthropic.APIStatusError):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.com"))
    return error_type("credential rejected", response=response, body=None)


class GenerateTracesTests(unittest.TestCase):
    def setUp(self):
        self.invoke_agent = Mock(return_value={"output": "answer"})
        self.agent_module = types.ModuleType("agent.agent")
        self.agent_module.invoke_agent = self.invoke_agent
        self.output = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.output))
        self.enterContext(patch.dict(sys.modules, {"agent.agent": self.agent_module}))
        self.enterContext(patch("scripts.generate_traces.time.sleep"))
        self.enterContext(patch("dotenv.load_dotenv"))
        self.enterContext(patch.dict(os.environ, {"LANGSMITH_API_KEY_GATEWAY": "gateway-test"}))
        self.enterContext(patch.dict(sys.modules))
        sys.modules.pop("utils.models", None)
        self.enterContext(patch.object(generate_traces, "QUERIES", [
            {"question": "single one"}, {"question": "single two"},
        ]))
        self.enterContext(patch.object(generate_traces, "THREADS", [
            {"name": "thread", "turns": [{"question": "turn one"}, {"question": "turn two"}]},
        ]))

    def assert_setup_failure(self):
        with self.assertRaises(SystemExit) as caught:
            generate_traces.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("Setup failed:", self.output.getvalue())
        self.invoke_agent.assert_not_called()

    def test_missing_gateway_credentials_during_initialization(self):
        os.environ.pop("LANGSMITH_API_KEY_GATEWAY")
        self.assert_setup_failure()
        self.assertIn("LANGSMITH_API_KEY_GATEWAY", self.output.getvalue())

    def test_agent_import_failure_is_setup_failure(self):
        with patch.dict(sys.modules, {"agent.agent": None}):
            self.assert_setup_failure()

    def test_failed_preflight_prevents_traced_queries(self):
        for error in (status_error(424), RuntimeError("endpoint unavailable")):
            with self.subTest(error=error), patch.object(ChatAnthropic, "invoke", side_effect=error):
                self.assert_setup_failure()
                self.assertIn(str(error), self.output.getvalue())

    def test_preflight_uses_gateway_model_without_anthropic_key_and_allows_both_loops(self):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["ANTHROPIC_BASE_URL"] = "https://unrelated.example.com"
        os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = "unrelated-model"
        tracing_states = []

        def invoke(model, message, **kwargs):
            tracing_states.append(get_tracing_context()["enabled"])
            self.assertEqual(model.model, "claude-sonnet-4-6")
            self.assertEqual(model.anthropic_api_url, "https://gateway.smith.langchain.com/anthropic")
            self.assertEqual(model.anthropic_api_key.get_secret_value(), "gateway-test")
            self.assertEqual(kwargs["max_tokens"], 1)
            return Mock()

        with patch.object(ChatAnthropic, "invoke", autospec=True, side_effect=invoke):
            generate_traces.main()
        self.assertEqual(tracing_states, [False])
        self.assertEqual(self.invoke_agent.call_count, 4)
        calls = self.invoke_agent.call_args_list
        self.assertNotIn("thread_id", calls[0].kwargs)
        self.assertEqual(calls[2].kwargs["thread_id"], calls[3].kwargs["thread_id"])
        self.assertIn("Done.", self.output.getvalue())

    def test_fatal_credentials_stop_each_loop(self):
        errors = [
            status_error(401, anthropic.AuthenticationError),
            status_error(403, anthropic.PermissionDeniedError),
            *(status_error(status) for status in (401, 403, 424)),
            TypeError("Could not resolve authentication method. Expected api_key"),
        ]
        for loop in ("single", "thread"):
            for error in errors:
                with self.subTest(loop=loop, error=error):
                    self.invoke_agent.reset_mock(side_effect=True)
                    self.invoke_agent.side_effect = error
                    queries = [] if loop == "thread" else generate_traces.QUERIES
                    with patch.object(generate_traces, "QUERIES", queries), patch.object(
                        generate_traces, "_preflight", return_value=self.invoke_agent
                    ):
                        with self.assertRaises(SystemExit) as caught:
                            generate_traces.main()
                    self.assertEqual(caught.exception.code, 1)
                    self.assertEqual(self.invoke_agent.call_count, 1)
                    self.assertIn(str(error), self.output.getvalue())

    def test_unrelated_errors_recover_in_both_loops(self):
        for error in (RuntimeError("query failed"), TypeError("bad output"), status_error(500)):
            with self.subTest(error=error):
                self.invoke_agent.reset_mock(side_effect=True)
                self.invoke_agent.side_effect = [error, {"output": "answer"}, error, {"output": "answer"}]
                with patch.object(generate_traces, "_preflight", return_value=self.invoke_agent):
                    generate_traces.main()
                self.assertEqual(self.invoke_agent.call_count, 4)
                self.assertIn(str(error), self.output.getvalue())
                self.assertIn("Done.", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
