"""Filesystem checks run without torch, PEFT, or a GPU."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "scripts/xtrainer/merge_xvla_lora.py"
SPEC = importlib.util.spec_from_file_location("merge_xvla", MODULE)
merge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(merge)


class MergePathsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.adapter = self.root / "adapter"
        self.output = self.root / "merged"
        self.base.mkdir()
        self.adapter.mkdir()
        for name in ("config.json", "model.safetensors"):
            (self.base / name).touch()
        for name in ("config.json", "adapter_config.json", "adapter_model.safetensors",
                     "policy_preprocessor.json", "policy_postprocessor.json", "xvla_base_manifest.json"):
            (self.adapter / name).touch()

    def test_valid_paths_do_not_create_output(self):
        merge.validate_paths(self.base, self.adapter, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_output_rejected(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exist"):
            merge.validate_paths(self.base, self.adapter, self.output)

    def test_nested_output_rejected(self):
        for source in (self.base, self.adapter):
            with self.assertRaisesRegex(ValueError, "overlap"):
                merge.validate_paths(self.base, self.adapter, source / "merged")

    def test_missing_processor_rejected(self):
        (self.adapter / "policy_preprocessor.json").unlink()
        with self.assertRaisesRegex(ValueError, "Required file"):
            merge.validate_paths(self.base, self.adapter, self.output)

    def test_missing_base_manifest_rejected(self):
        (self.adapter / "xvla_base_manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "Required file"):
            merge.validate_paths(self.base, self.adapter, self.output)


if __name__ == "__main__":
    unittest.main()
