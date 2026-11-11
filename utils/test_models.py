import os
import unittest
from unittest.mock import patch

from utils.models import get_model, get_model_config


class ModelSelectionTests(unittest.TestCase):
    def test_selection_and_gateway_settings(self):
        for override in (None, "", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"):
            with (
                self.subTest(override=override),
                patch.dict(os.environ, {"LANGSMITH_API_KEY_GATEWAY": "test-key"}),
            ):
                if override is None:
                    os.environ.pop("CHAT_LANGCHAIN_LITE_MODEL", None)
                else:
                    os.environ["CHAT_LANGCHAIN_LITE_MODEL"] = override
                expected = override or "claude-haiku-4-5-20251001"
                config = get_model_config()
                model = get_model()
                self.assertEqual(
                    config,
                    {
                        "model": expected,
                        "provider": "anthropic",
                        "base_url": "https://gateway.smith.langchain.com/anthropic",
                    },
                )
                self.assertEqual(model.model, expected)
                self.assertEqual(model.anthropic_api_url, config["base_url"])
                self.assertEqual(model.anthropic_api_key.get_secret_value(), "test-key")
                self.assertEqual(model.max_tokens, 300)
                self.assertEqual(model.temperature, 0)


if __name__ == "__main__":
    unittest.main()
