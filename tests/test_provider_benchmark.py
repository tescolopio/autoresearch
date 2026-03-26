import json
import pathlib
import tempfile
import unittest

from provider_benchmark import ProviderConfig, load_config, run_suite, save_outputs


class ProviderBenchmarkTests(unittest.TestCase):
    def test_load_config_parses_providers_and_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "providers": [{"name": "mock-a", "type": "mock", "mock_responses": {"task-1": "42"}}],
                        "tasks": [{"id": "task-1", "prompt": "What is the answer?", "must_contain": ["42"]}],
                    }
                ),
                encoding="utf-8",
            )
            providers, tasks = load_config(config_path)
        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0].name, "mock-a")
        self.assertEqual(tasks[0]["id"], "task-1")

    def test_run_suite_supports_mock_provider(self):
        providers = [ProviderConfig(name="mock-a", type="mock", mock_responses={"task-1": "42"})]
        tasks = [{"id": "task-1", "prompt": "What is the answer?", "must_contain": ["42"]}]
        frame = run_suite(providers, tasks)
        self.assertEqual(len(frame), 1)
        self.assertTrue(bool(frame.iloc[0]["pass"]))
        self.assertEqual(frame.iloc[0]["provider"], "mock-a")

    def test_save_outputs_writes_artifacts(self):
        providers = [ProviderConfig(name="mock-a", type="mock", mock_responses={"task-1": "42"})]
        tasks = [{"id": "task-1", "prompt": "What is the answer?", "must_contain": ["42"]}]
        frame = run_suite(providers, tasks)
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = save_outputs(frame, tmpdir, "Provider Comparison")
            self.assertTrue(outputs["jsonl"].exists())
            self.assertTrue(outputs["csv"].exists())
            self.assertTrue(outputs["markdown"].exists())
            self.assertTrue(outputs["chart"].exists())


if __name__ == "__main__":
    unittest.main()