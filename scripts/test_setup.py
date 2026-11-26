import copy
import unittest
from unittest.mock import Mock, patch

from scripts import setup


class ScopeEvaluatorSetupTests(unittest.TestCase):
    def setUp(self):
        self.rules = [
            {
                "id": ev["rule_id"],
                "session_id": setup.SCOPE_PROJECT_ID,
                "display_name": ev["feedback_key"].title(),
                "sampling_rate": 0.75,
                "evaluators": [{"structured": {
                    "playground_settings_id": f"playground-{index}",
                    "model": {"id": ["existing", "model"], "kwargs": {"temperature": 0.2}},
                    "schema": {"properties": {ev["feedback_key"]: {"type": "integer"}}},
                    "prompt": [["system", "old rubric"]],
                    "variable_mapping": {"output": "output"},
                }}],
            }
            for index, ev in enumerate(setup.SCOPE_EVALUATORS)
        ]

    def test_updates_existing_rules_and_preserves_configuration(self):
        original = copy.deepcopy(self.rules)
        with patch.object(setup.requests, "get", return_value=Mock(json=lambda: self.rules)), \
                patch.object(setup.requests, "patch", return_value=Mock()) as update, \
                patch.object(setup.requests, "post") as create, \
                patch.object(setup.requests, "delete") as delete:
            for _ in range(2):
                ids = setup.update_scope_evaluators("test-key", setup.SCOPE_PROJECT_ID)
                self.assertEqual(ids, [rule["id"] for rule in self.rules])
            self.assertEqual(update.call_count, 4)
            for call, rule in zip(update.call_args_list[:2], original):
                self.assertTrue(call.args[0].endswith(rule["id"]))
                payload = call.kwargs["json"]
                self.assertEqual(payload["display_name"], rule["display_name"])
                self.assertEqual(payload["sampling_rate"], rule["sampling_rate"])
                structured = payload["evaluators"][0]["structured"]
                preserved = copy.deepcopy(structured)
                for key in ("prompt", "variable_mapping"):
                    preserved[key] = rule["evaluators"][0]["structured"][key]
                self.assertEqual(preserved, rule["evaluators"][0]["structured"])
                self.assertEqual(structured["variable_mapping"], {"input": "input", "output": "output"})
                rubric = structured["prompt"][0][1]
                for category in (
                    "LangChain", "LangGraph", "LangSmith", "Deep Agents",
                    "integrations", "CI/CD", "Kubernetes/Jaeger", "Django",
                    "Python utilities", "transformer math", "business plans/GTM",
                    "model or vector-database comparisons",
                ):
                    self.assertIn(category, rubric)
                for exemption in (
                    "Score 1 for a polite decline", "redirects",
                    "Offering an in-scope version", "asking for code",
                    "suggesting other resources", "score 0", "incorrect",
                    "incomplete", "truncated",
                ):
                    self.assertIn(exemption, rubric)
                human = structured["prompt"][1][1]
                rendered = human.replace("{{input}}", "request text").replace("{{output}}", "response text")
                self.assertIn("request text", rendered)
                self.assertIn("response text", rendered)
            create.assert_not_called()
            delete.assert_not_called()
        self.assertEqual(self.rules, original)

    def test_missing_rule_fails_before_any_updates(self):
        with patch.object(setup.requests, "get", return_value=Mock(json=lambda: self.rules[:1])), \
                patch.object(setup.requests, "patch") as update:
            with self.assertRaises(KeyError):
                setup.update_scope_evaluators("test-key", setup.SCOPE_PROJECT_ID)
            update.assert_not_called()

    def test_unexpected_schema_fails_before_any_updates(self):
        self.rules[1]["evaluators"][0]["structured"]["schema"]["properties"] = {"different": {}}
        with patch.object(setup.requests, "get", return_value=Mock(json=lambda: self.rules)), \
                patch.object(setup.requests, "patch") as update:
            with self.assertRaises(ValueError):
                setup.update_scope_evaluators("test-key", setup.SCOPE_PROJECT_ID)
            update.assert_not_called()

    def test_other_projects_are_unchanged(self):
        with patch.object(setup.requests, "get") as read:
            self.assertEqual(setup.update_scope_evaluators("test-key", "another-project"), [])
            read.assert_not_called()

    def test_cleanup_preserves_scope_rules_and_deletes_original_rules(self):
        for rule in self.rules:
            rule["display_name"] = "chat-lc-lite-demo-existing"
        original_rule = {"id": "original", "display_name": "security_advice"}
        with patch.object(setup.requests, "get", side_effect=[
            Mock(status_code=200, json=lambda: self.rules + [original_rule]),
            Mock(status_code=200, json=lambda: {"evaluators": []}),
            Mock(status_code=200, json=lambda: {"evaluators": []}),
        ]), patch.object(setup.requests, "delete") as delete:
            setup.delete_existing_evaluators("test-key")
            self.assertEqual(delete.call_count, 1)
            self.assertTrue(delete.call_args.args[0].endswith("/original"))


if __name__ == "__main__":
    unittest.main()
