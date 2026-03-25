import ast
import os
import pathlib
import tempfile
import types
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "train.py"
PREPARE_PATH = REPO_ROOT / "prepare.py"


def load_train_symbols():
    source = TRAIN_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(TRAIN_PATH))
    wanted_assignments = {"RESULTS_HEADER"}
    wanted_functions = {
        "compute_objective_signature",
        "verify_objective_signature",
        "ensure_results_tsv",
        "append_results_tsv",
    }
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if targets & wanted_assignments:
                selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            selected_nodes.append(node)

    module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace = {
        "os": os,
        "hmac": __import__("hmac"),
        "hashlib": __import__("hashlib"),
    }
    exec(compile(module, filename=str(TRAIN_PATH), mode="exec"), namespace, namespace)
    return types.SimpleNamespace(**namespace)


class BitNetCpuPocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = load_train_symbols()
        cls.train_source = TRAIN_PATH.read_text(encoding="utf-8")
        cls.prepare_source = PREPARE_PATH.read_text(encoding="utf-8")
        cls.train_tree = ast.parse(cls.train_source, filename=str(TRAIN_PATH))
        cls.prepare_tree = ast.parse(cls.prepare_source, filename=str(PREPARE_PATH))

    def get_function_def(self, tree, name):
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        self.fail(f"Function {name} not found")

    def test_train_exposes_cpu_bitnet_cli_and_bitlinear_mode(self):
        self.assertIn("--cpu-bitnet-poc", self.train_source)
        self.assertIn('choices=["dense", "bitlinear"]', self.train_source)
        self.assertIn("class BitLinear", self.train_source)

    def test_prepare_supports_device_aware_dataloader_and_eval(self):
        make_dataloader = self.get_function_def(self.prepare_tree, "make_dataloader")
        evaluate_bpb = self.get_function_def(self.prepare_tree, "evaluate_bpb")
        self.assertEqual([arg.arg for arg in make_dataloader.args.args][-1], "device")
        self.assertEqual(ast.literal_eval(make_dataloader.args.defaults[-1]), "cuda")
        self.assertEqual([arg.arg for arg in evaluate_bpb.args.args][-1], "device")
        self.assertEqual(ast.literal_eval(evaluate_bpb.args.defaults[-1]), "cuda")
        self.assertIn('device_buffer is not None', self.prepare_source)

    def test_signature_verification_accepts_valid_hmac(self):
        objective = "Minimize energy while maintaining accuracy"
        secret = "cpu-bitnet-demo"
        signature = self.train.compute_objective_signature(objective, secret)
        self.assertTrue(self.train.verify_objective_signature(objective, signature, secret, require_signature=True))

    def test_signature_verification_rejects_invalid_hmac(self):
        with self.assertRaises(RuntimeError):
            self.train.verify_objective_signature("goal", "bad-signature", "secret", require_signature=True)

    def test_signature_verification_rejects_partial_signature_data(self):
        with self.assertRaises(RuntimeError):
            self.train.verify_objective_signature("goal", "", "secret", require_signature=False)

    def test_results_tsv_includes_cpu_metrics_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.tsv")
            self.train.append_results_tsv(
                path,
                {
                    "commit": "abc1234",
                    "val_bpb": 1.234567,
                    "memory_gb": 0.4,
                    "status": "candidate",
                    "description": "cpu\tbitnet poc\rwith newline\nsuffix",
                    "device": "cpu",
                    "linear_impl": "bitlinear",
                    "signature_verified": True,
                    "energy_j_per_token": 0.000123456,
                    "tokens_per_second": 42.0,
                },
            )
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f]
            self.assertEqual(len(lines), 2)
            self.assertIn("energy_j_per_token", lines[0])
            self.assertIn("tokens_per_second", lines[0])
            self.assertIn("bitlinear", lines[1])
            self.assertIn("yes", lines[1])
            self.assertNotIn("\n", lines[1])
            self.assertNotIn("\tbitnet", lines[1])
            self.assertNotIn("\r", lines[1])


if __name__ == "__main__":
    unittest.main()
