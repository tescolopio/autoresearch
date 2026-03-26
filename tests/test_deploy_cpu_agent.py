import json
import pathlib
import tempfile
import unittest

from deploy_cpu_agent import DeploymentConfig, build_manifest, cache_status, deploy_agent, resolve_mode


class DeployCpuAgentTests(unittest.TestCase):
    def test_resolve_mode_prefers_mock_when_cache_missing(self):
        self.assertEqual(resolve_mode("auto", {"ready": False}), "mock")
        self.assertEqual(resolve_mode("auto", {"ready": True}), "train")

    def test_cache_status_uses_expected_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status = cache_status(tmpdir)
            self.assertFalse(status["ready"])
            data_dir = pathlib.Path(status["data_dir"])
            tokenizer_dir = pathlib.Path(status["tokenizer_dir"])
            data_dir.mkdir(parents=True)
            tokenizer_dir.mkdir(parents=True)
            self.assertFalse(cache_status(tmpdir)["ready"])
            (data_dir / "shard_00000.parquet").write_text("placeholder", encoding="utf-8")
            (tokenizer_dir / "tokenizer.pkl").write_text("placeholder", encoding="utf-8")
            (tokenizer_dir / "token_bytes.pt").write_text("placeholder", encoding="utf-8")
            status = cache_status(tmpdir)
            self.assertTrue(status["ready"])

    def test_build_manifest_marks_cpu_only_and_mock_next_step(self):
        config = DeploymentConfig(repo_root=".", mode="mock")
        manifest = build_manifest(config, {"ready": False}, "mock")
        self.assertTrue(manifest["cpu_only"])
        self.assertEqual(manifest["deployment_mode"], "mock")
        self.assertIn("prepare.py", manifest["next_step"])

    def test_deploy_agent_writes_report_in_mock_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DeploymentConfig(repo_root=tmpdir, mode="mock", agent_name="dev-agent", iterations=1)
            paths, report = deploy_agent(config)
            self.assertTrue(paths["manifest"].exists())
            self.assertTrue(paths["report"].exists())
            saved_report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(saved_report["manifest"]["deployment_mode"], "mock")
            self.assertEqual(report["best"]["metrics"]["device"], "cpu")
            self.assertEqual(report["best"]["metrics"]["linear_impl"], "bitlinear")


if __name__ == "__main__":
    unittest.main()