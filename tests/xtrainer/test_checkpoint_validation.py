import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[2] / "deploy/xtrainer/checkpoint_validation.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_validation", MODULE)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class DeploymentCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def check(self):
        return module.validate_deployment_checkpoint(str(self.root))

    def merged(self):
        for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            (self.root / name).touch()
        (self.root / "config.json").write_text('{"use_peft": false}', encoding="utf-8")
        (self.root / "tokenizer").mkdir()
        report = {"validated": True, "merge_max_abs_error": 0.001, "reload_max_abs_error": 0}
        (self.root / "merge_report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_legacy_full_model_unchanged(self):
        self.assertFalse(self.check())

    def test_valid_merge(self):
        self.merged()
        self.assertTrue(self.check())

    def test_adapter_rejected(self):
        (self.root / "adapter_config.json").touch()
        with self.assertRaisesRegex(ValueError, "standalone PEFT"):
            self.check()

    def test_failed_export_rejected(self):
        self.merged()
        (self.root / "EXPORT_FAILED.txt").touch()
        with self.assertRaisesRegex(ValueError, "failed export"):
            self.check()

    def test_incomplete_export_rejected(self):
        (self.root / "xvla_base_manifest.json").touch()
        with self.assertRaisesRegex(ValueError, "merge_report"):
            self.check()

    def test_missing_processors_rejected(self):
        self.merged()
        (self.root / "policy_preprocessor.json").unlink()
        with self.assertRaisesRegex(ValueError, "preprocessor"):
            self.check()

    def test_peft_config_rejected(self):
        self.merged()
        (self.root / "config.json").write_text('{"use_peft": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "use_peft"):
            self.check()

    def test_invalid_report_rejected(self):
        self.merged()
        (self.root / "merge_report.json").write_text('{"validated": false}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "successful"):
            self.check()


if __name__ == "__main__":
    unittest.main()
