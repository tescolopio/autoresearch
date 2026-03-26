import pathlib
import tempfile
import unittest

from benchmark_compare import benchmark_paths, build_mock_gpu_summary, materialize_mock_gpu_baseline


class BenchmarkCompareTests(unittest.TestCase):
    def test_benchmark_paths_creates_expected_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = benchmark_paths(pathlib.Path(tmpdir) / "bench")
            self.assertTrue(paths["cpu_json"].parent.exists())
            self.assertEqual(paths["cpu_json"].name, "cpu_bitnet.json")
            self.assertEqual(paths["gpu_tsv"].name, "gpu_dense.tsv")

    def test_build_mock_gpu_summary_has_cuda_metrics(self):
        summary = build_mock_gpu_summary()
        self.assertEqual(summary["device"], "cuda")
        self.assertGreater(summary["tokens_per_second"], 0.0)
        self.assertGreater(summary["avg_gpu_util_percent"], 0.0)

    def test_materialize_mock_gpu_baseline_writes_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = materialize_mock_gpu_baseline(tmpdir)
            self.assertTrue(paths["gpu_json"].exists())
            self.assertEqual(paths["gpu_json"].stem, "gpu_dense")


if __name__ == "__main__":
    unittest.main()