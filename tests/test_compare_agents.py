import json
import pathlib
import tempfile
import unittest

from compare_agents import build_skeptic_summary, collect_rows, generate_report, to_frame


class CompareAgentsTests(unittest.TestCase):
    def test_collect_rows_supports_tsv_and_summary_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            tsv_path = tmp / "cpu.tsv"
            tsv_path.write_text(
                "commit\tval_bpb\tmemory_gb\tstatus\tdescription\tdevice\tlinear_impl\tsignature_verified\tenergy_j_per_token\ttokens_per_second\tavg_cpu_process_percent\tavg_cpu_load_percent\tavg_gpu_util_percent\tavg_gpu_mem_used_mb\treliability_score\tcapability_score\ttask_eval_score\n"
                "abc1234\t1.100000\t0.4\tkeep\tcpu baseline\tcpu\tbitlinear\tyes\t0.028000000\t6.2\t85.0\t65.0\t0.0\t0.0\t0.9500\t0.7200\t0.8580\n",
                encoding="utf-8",
            )
            summary_path = tmp / "gpu.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "commit": "def5678",
                        "val_bpb": 0.95,
                        "memory_gb": 12.0,
                        "status": "keep",
                        "description": "gpu dense baseline",
                        "device": "cuda",
                        "linear_impl": "dense",
                        "signature_verified": False,
                        "energy_j_per_token": 0.19,
                        "tokens_per_second": 18.0,
                        "avg_cpu_process_percent": 10.0,
                        "avg_cpu_load_percent": 12.0,
                        "avg_gpu_util_percent": 76.0,
                        "avg_gpu_mem_used_mb": 12000.0,
                        "reliability_score": 0.0,
                        "capability_score": 0.0,
                        "task_eval_score": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            rows = collect_rows([tsv_path], [summary_path], [])
            frame = to_frame(rows)
        self.assertEqual(len(frame), 2)
        self.assertSetEqual(set(frame["device"]), {"cpu", "cuda"})
        self.assertIn("task_eval_score", frame.columns)

    def test_generate_report_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            frame = to_frame(
                [
                    {
                        "commit": "abc1234",
                        "val_bpb": 1.1,
                        "memory_gb": 0.4,
                        "status": "keep",
                        "description": "cpu baseline",
                        "device": "cpu",
                        "linear_impl": "bitlinear",
                        "signature_verified": True,
                        "energy_j_per_token": 0.028,
                        "tokens_per_second": 6.2,
                        "avg_cpu_process_percent": 85.0,
                        "avg_cpu_load_percent": 65.0,
                        "avg_gpu_util_percent": 0.0,
                        "avg_gpu_mem_used_mb": 0.0,
                        "reliability_score": 0.95,
                        "capability_score": 0.72,
                        "task_eval_score": 0.858,
                        "source_path": "cpu.tsv",
                    },
                    {
                        "commit": "def5678",
                        "val_bpb": 0.95,
                        "memory_gb": 12.0,
                        "status": "keep",
                        "description": "gpu dense baseline",
                        "device": "cuda",
                        "linear_impl": "dense",
                        "signature_verified": False,
                        "energy_j_per_token": 0.19,
                        "tokens_per_second": 18.0,
                        "avg_cpu_process_percent": 10.0,
                        "avg_cpu_load_percent": 12.0,
                        "avg_gpu_util_percent": 76.0,
                        "avg_gpu_mem_used_mb": 12000.0,
                        "reliability_score": 0.0,
                        "capability_score": 0.0,
                        "task_eval_score": 0.0,
                        "source_path": "gpu.json",
                    },
                ]
            )
            outputs = generate_report(frame, pathlib.Path(tmpdir) / "report", "Comparison")
            self.assertTrue(outputs["csv"].exists())
            self.assertTrue(outputs["markdown"].exists())
            self.assertTrue(outputs["skeptic_markdown"].exists())
            self.assertTrue(outputs["throughput_png"].exists())
            self.assertTrue(outputs["resource_png"].exists())
            self.assertTrue(outputs["efficiency_png"].exists())

    def test_build_skeptic_summary_calls_out_limits(self):
        frame = to_frame(
            [
                {
                    "commit": "abc1234",
                    "val_bpb": 1.1,
                    "memory_gb": 0.4,
                    "status": "keep",
                    "description": "cpu baseline",
                    "device": "cpu",
                    "linear_impl": "bitlinear",
                    "signature_verified": True,
                    "energy_j_per_token": 0.028,
                    "tokens_per_second": 6.2,
                    "avg_cpu_process_percent": 85.0,
                    "avg_cpu_load_percent": 65.0,
                    "avg_gpu_util_percent": 0.0,
                    "avg_gpu_mem_used_mb": 0.0,
                    "reliability_score": 0.95,
                    "capability_score": 0.72,
                    "task_eval_score": 0.858,
                    "source_path": "cpu.tsv",
                },
                {
                    "commit": "def5678",
                    "val_bpb": 0.95,
                    "memory_gb": 12.0,
                    "status": "keep",
                    "description": "gpu dense baseline",
                    "device": "cuda",
                    "linear_impl": "dense",
                    "signature_verified": False,
                    "energy_j_per_token": 0.19,
                    "tokens_per_second": 18.0,
                    "avg_cpu_process_percent": 10.0,
                    "avg_cpu_load_percent": 12.0,
                    "avg_gpu_util_percent": 76.0,
                    "avg_gpu_mem_used_mb": 12000.0,
                    "reliability_score": 0.0,
                    "capability_score": 0.0,
                    "task_eval_score": 0.0,
                    "source_path": "gpu.json",
                },
            ]
        )
        summary = build_skeptic_summary(frame, "Comparison")
        self.assertIn("What this does prove", summary)
        self.assertIn("What this does not prove by itself", summary)
        self.assertIn("task_eval_score", summary)


if __name__ == "__main__":
    unittest.main()