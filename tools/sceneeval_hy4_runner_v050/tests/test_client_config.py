import importlib.util
import unittest

from sceneeval_hy4.client_config import load_run_config


@unittest.skipIf(importlib.util.find_spec("yaml") is None, "PyYAML is unavailable")
class ClientConfigTests(unittest.TestCase):
    def test_frozen_yaml_resolves_to_approved_wire_configuration(self):
        config = load_run_config()
        self.assertEqual(
            config.endpoint,
            "http://infer-proxy-yb-test.production.polaris:8000/"
            "openapi/chat/completions",
        )
        self.assertEqual(
            config.configured_model,
            "openai/Hy4-T3-A49B-DSA-1M-SFT0730-Opus5",
        )
        self.assertEqual(
            config.wire_model,
            "Hy4-T3-A49B-DSA-1M-SFT0730-Opus5",
        )
        self.assertEqual(config.api_key, "EMPTY")
        self.assertEqual(config.timeout_seconds, 1800.0)
        self.assertEqual(config.max_retries, 2)
        self.assertEqual(config.temperature, 0.9)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.top_k, -1)
        self.assertEqual(config.max_tokens, 65536)
        self.assertEqual(config.repetition_penalty, 1.0)
        self.assertEqual(config.reasoning_effort, "high")
        self.assertTrue(config.preserved_thinking)
        self.assertEqual(config.strategy_type, "ConsistentHash")


if __name__ == "__main__":
    unittest.main()
