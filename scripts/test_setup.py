"""Tests for online evaluator setup without external services."""

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import setup, run_evals


def response(body):
    return Mock(status_code=200, ok=True, json=Mock(return_value=body))


class OnlineEvaluatorSetupTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.env = patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "LANGSMITH_API_KEY": "api-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.get = patch.object(setup.requests, "get").start()
        self.post = patch.object(setup.requests, "post").start()
        self.delete = patch.object(setup.requests, "delete").start()
        self.client = patch("langsmith.Client").start().return_value
        patch("langchain_anthropic.ChatAnthropic").start().return_value.to_json.return_value = {}
        patch.object(setup.time, "sleep").start()
        self.addCleanup(patch.stopall)
        self.client.list_projects.return_value = [SimpleNamespace(name=setup.PROJECT_NAME, id="project-id")]
        self.get.side_effect = lambda url, **kwargs: response(
            [{"key": "ANTHROPIC_API_KEY"}] if url.endswith("secrets") else []
            if url.endswith("rules") else {"evaluators": []}
        )
        self.post.return_value = response({"id": "rule-id"})
        self.client.list_feedback.return_value = [
            SimpleNamespace(key=ev["feedback_key"], score=0, extra={}, comment=None)
            for ev in setup.EVALUATORS
        ]

    def rules(self):
        return [call.kwargs["json"] for call in self.post.call_args_list
                if call.args[0].endswith("/rules")]

    def test_configured_secret_creates_all_rules_and_checks_scores(self):
        with redirect_stdout(self.output):
            ids = setup.setup_online_evaluators("api-key")
        self.assertEqual(len(ids), len(setup.EVALUATORS))
        self.assertIn("smoke test passed", self.output.getvalue())
        for rule, ev in zip(self.rules(), setup.EVALUATORS):
            structured = rule["evaluators"][0]["structured"]
            self.assertEqual(structured["prompt"], [
                ["system", ev["system_prompt"]], ["human", "Agent response: {{output}}"]
            ])
            self.assertEqual(list(structured["schema"]["properties"]), [ev["feedback_key"]])

    def test_missing_secret_aborts_without_modifying_rules(self):
        self.get.side_effect = None
        self.get.return_value = response([])
        with redirect_stdout(self.output), self.assertRaisesRegex(RuntimeError, "Settings → Secrets"):
            setup.setup_online_evaluators("api-key")
        self.assertEqual(self.rules(), [])
        self.delete.assert_not_called()
        self.assertIn("Settings → Secrets", self.output.getvalue())

    def test_local_key_upload_is_verified_before_creating_rules(self):
        self.get.side_effect = [response([]), response([{"key": "ANTHROPIC_API_KEY"}]),
                                response([]), response({"evaluators": []}), response({"evaluators": []})]
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "local-key"}), \
                patch.object(setup.sys.stdin, "isatty", return_value=True), \
                patch("builtins.input", return_value="y"), redirect_stdout(self.output):
            setup.setup_online_evaluators("api-key")
        self.assertEqual(self.post.call_args_list[0].kwargs["json"],
                         [{"key": "ANTHROPIC_API_KEY", "value": "local-key"}])
        self.assertEqual(len(self.rules()), len(setup.EVALUATORS))
        self.assertNotIn("local-key", self.output.getvalue())

    def test_upload_without_available_secret_aborts(self):
        self.get.side_effect = None
        self.get.return_value = response([])
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "local-key"}), \
                patch.object(setup.sys.stdin, "isatty", return_value=True), \
                patch("builtins.input", return_value="y"), redirect_stdout(self.output), \
                self.assertRaisesRegex(RuntimeError, "No evaluator rules were created"):
            setup.setup_online_evaluators("api-key")
        self.assertEqual(self.rules(), [])
        self.delete.assert_not_called()

    def test_declined_upload_aborts_without_modifying_rules(self):
        self.get.side_effect = None
        self.get.return_value = response([])
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "local-key"}), \
                patch.object(setup.sys.stdin, "isatty", return_value=True), \
                patch("builtins.input", return_value="n"), redirect_stdout(self.output), \
                self.assertRaisesRegex(RuntimeError, "Settings → Secrets"):
            setup.setup_online_evaluators("api-key")
        self.post.assert_not_called()
        self.delete.assert_not_called()

    def test_secret_lookup_failure_aborts_without_modifying_rules(self):
        self.get.side_effect = setup.requests.HTTPError("Forbidden")
        with redirect_stdout(self.output), self.assertRaises(setup.requests.HTTPError):
            setup.setup_online_evaluators("api-key")
        self.post.assert_not_called()
        self.delete.assert_not_called()

    def test_error_feedback_fails_setup(self):
        self.client.list_feedback.return_value = [SimpleNamespace(
            key="factual_accuracy", score=None, extra={"error": True},
            comment="Anthropic authentication failed",
        )]
        with redirect_stdout(self.output), self.assertRaisesRegex(RuntimeError, "Anthropic authentication failed"):
            setup.setup_online_evaluators("api-key")
        self.assertNotIn("Every future trace", self.output.getvalue())

    def test_missing_feedback_times_out(self):
        self.client.list_feedback.return_value = []
        with patch.object(setup.time, "monotonic", side_effect=[0, 1, 121]), \
                redirect_stdout(self.output), self.assertRaisesRegex(RuntimeError, "timed out"):
            setup.setup_online_evaluators("api-key")
        self.assertNotIn("Every future trace", self.output.getvalue())

    def test_legacy_entry_point_checks_credentials(self):
        self.get.side_effect = None
        self.get.return_value = response([])
        with redirect_stdout(self.output), self.assertRaisesRegex(RuntimeError, "Settings → Secrets"):
            run_evals.setup_online_eval()
        self.assertEqual(self.rules(), [])


if __name__ == "__main__":
    unittest.main()
