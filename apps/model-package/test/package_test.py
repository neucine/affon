import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("model_package", Path(__file__).parents[1] / "package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)
GRAPH = package.ROOT / "packages/@affon/onnx/test/fixtures"
LLAMA = package.ROOT / "packages/@affon/huggingface/test/fixtures/llama"
AFFON = str(package.ROOT / "zig-out/bin/affon")


def flatten(value):
    return [v for child in value for v in flatten(child)] if isinstance(value, list) else [value]


class PackageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def build(self, source=GRAPH, runtime="affon/prepared-onnx/v1", name="package"):
        destination = self.root / name
        files = package.ADAPTERS[runtime]
        package.pack(source, destination, runtime, files, files[0])
        return destination

    def test_pack_is_reproducible_and_relocatable(self):
        source = self.root / "source"
        shutil.copytree(GRAPH, source)
        first = self.build(source, name="one")
        second = self.build(source, name="two")
        self.assertEqual((first / "manifest.json").read_bytes(), (second / "manifest.json").read_bytes())
        shutil.rmtree(source)
        moved = self.root / "moved"
        first.rename(moved)
        self.assertEqual(package.verify(moved), package.verify(second))

    def test_convolution_graph_matches_independent_onnx_reference(self):
        directory = self.build()
        reference = json.loads((GRAPH / "reference.json").read_text())
        response = package.execute(directory, {"inputs": {"x": reference["input"]}}, engine="affon", executable=AFFON)
        self.assertEqual(set(response["outputs"]), set(reference["expected"]))
        for name, expected in reference["expected"].items():
            actual, expected = flatten(response["outputs"][name]["data"]), flatten(expected)
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assertLessEqual(abs(a - b), 1e-5 + 1e-5 * abs(b))
        self.assertEqual(response["outputs"]["result"]["shape"], [1, 2])

    def test_llama_generation_matches_independent_pytorch_reference(self):
        directory = self.build(LLAMA, "affon/llama-tokens/v1")
        reference = json.loads((LLAMA / "reference.json").read_text())
        response = package.execute(directory, {"input_ids": reference["ids"], "max_new_tokens": 4}, engine="affon", executable=AFFON)
        self.assertEqual(response["token_ids"], reference["generated"])

    def test_altered_or_missing_artifact_never_launches_runtime(self):
        directory = self.build()
        weights = directory / "weights.safetensors"
        data = bytearray(weights.read_bytes())
        data[-1] ^= 1  # Same length; must check the digest as well as size.
        weights.write_bytes(data)
        with patch.object(package.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "Integrity"):
                package.execute(directory, {}, engine="affon")
            weights.unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                package.execute(directory, {}, engine="affon")
            launch.assert_not_called()

    def test_unknown_adapter_can_be_packaged_but_cannot_execute(self):
        directory = self.root / "custom"
        package.pack(GRAPH, directory, "example/custom/v1", ["graph.json"], "graph.json")
        self.assertEqual(package.verify(directory)["contract"], "example/custom/v1")
        with patch.object(package.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "Unsupported model contract"):
                package.execute(directory, {}, engine="affon")
            launch.assert_not_called()

    def test_unknown_schema_invalid_paths_and_duplicates_are_rejected(self):
        base = package.read_manifest(self.build())
        for path in ["../weights", "/tmp/weights", "nested/weights", "manifest.json"]:
            candidate = copy.deepcopy(base)
            candidate["artifacts"][0]["path"] = path
            with self.assertRaises(ValueError):
                package.validate(candidate)
        candidate = copy.deepcopy(base)
        candidate["artifacts"].append(candidate["artifacts"][0])
        with self.assertRaises(ValueError):
            package.validate(candidate)
        candidate = {**base, "schema": "future/v99"}
        with self.assertRaises(ValueError):
            package.validate(candidate)

    def test_incomplete_adapter_files_rejected_before_launch(self):
        directory = self.root / "incomplete"
        package.pack(GRAPH, directory, "affon/prepared-onnx/v1", ["graph.json"], "graph.json")
        with patch.object(package.subprocess, "run") as launch:
            with self.assertRaisesRegex(ValueError, "file contract"):
                package.execute(directory, {}, engine="affon")
            launch.assert_not_called()

    def test_failed_pack_leaves_no_partial_destination_or_overwrites(self):
        destination = self.root / "bad"
        with self.assertRaises(ValueError):
            package.pack(GRAPH, destination, "example/v1", ["absent.bin"], "absent.bin")
        self.assertFalse(destination.exists())
        valid = self.build()
        original = (valid / "manifest.json").read_bytes()
        with self.assertRaises(ValueError):
            package.pack(GRAPH, valid, "example/v1", ["graph.json"], "graph.json")
        self.assertEqual((valid / "manifest.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
