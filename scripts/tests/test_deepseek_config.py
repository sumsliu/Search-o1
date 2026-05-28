import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class TestDeepSeekConfig(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            "DEEPSEEK_MODEL": "flash",
            "DEEPSEEK_MODEL_FLASH": "deepseek-v4-flash",
            "DEEPSEEK_MODEL_PRO": "deepseek-v4-pro",
            "DEEPSEEK_THINKING_ENABLED": "true",
            "DEEPSEEK_REASONING_EFFORT": "high",
        },
        clear=False,
    )
    def test_get_deepseek_model_variants(self):
        import importlib
        import config

        importlib.reload(config)
        self.assertEqual(config.get_deepseek_model(), "deepseek-v4-flash")
        self.assertEqual(config.get_deepseek_model("flash"), "deepseek-v4-flash")
        self.assertEqual(config.get_deepseek_model("pro"), "deepseek-v4-pro")
        self.assertEqual(config.require_deepseek_api_key(), "test-key")

    @patch.dict(
        os.environ,
        {"DEEPSEEK_MODEL": "pro", "DEEPSEEK_MODEL_FLASH": "deepseek-v4-flash", "DEEPSEEK_MODEL_PRO": "deepseek-v4-pro"},
        clear=False,
    )
    def test_default_model_pro(self):
        import importlib
        import config

        importlib.reload(config)
        self.assertEqual(config.get_deepseek_model(), "deepseek-v4-pro")


class TestSearchO1ApiHelpers(unittest.TestCase):
    def test_extract_between_search_query(self):
        from run_search_o1_api import (
            BEGIN_SEARCH_QUERY,
            END_SEARCH_QUERY,
            extract_between,
            truncate_prev_reasoning,
        )

        text = f"think {BEGIN_SEARCH_QUERY}test query{END_SEARCH_QUERY}"
        self.assertEqual(extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY), "test query")

        lines = [f"line {i}" for i in range(10)]
        lines[5] = f"{BEGIN_SEARCH_QUERY}q{END_SEARCH_QUERY}"
        output = "\n".join(lines)
        truncated = truncate_prev_reasoning(output)
        self.assertIn(BEGIN_SEARCH_QUERY, truncated)


if __name__ == "__main__":
    unittest.main()
