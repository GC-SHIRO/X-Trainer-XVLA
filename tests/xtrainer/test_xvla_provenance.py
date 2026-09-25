"""Run with unittest; no model runtime or checkpoint download required."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

MODULE = Path(__file__).resolve().parents[2] / "src/lerobot/common/xvla_provenance.py"
SPEC = importlib.util.spec_from_file_location("xvla_provenance", MODULE)
provenance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provenance)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "config.json").write_text('{"type": "xvla"}', encoding="utf-8")
        (self.base / "model.safetensors").write_bytes(b"test-weight-bytes")
        self.adapter = self.root / "adapter"
        self.manifest = {
            "schema_version": 1, "base": provenance.fingerprint_base(self.base),
        }
        provenance.save_manifest(SimpleNamespace(_xvla_base_manifest=self.manifest), self.adapter)
        self.manifest["artifacts"] = {}

    def test_save_roundtrip(self):
        self.assertEqual(provenance.verify_manifest(self.adapter, self.base), self.manifest)
        saved = json.loads((self.adapter / provenance.MANIFEST).read_text(encoding="utf-8"))
        self.assertIn("sha256", saved["base"]["files"]["model.safetensors"])

    def test_relocated_identical_base(self):
        moved = self.root / "moved"
        shutil.copytree(self.base, moved)
        self.assertEqual(provenance.verify_manifest(self.adapter, moved), self.manifest)

    def test_changed_weights_rejected(self):
        (self.base / "model.safetensors").write_bytes(b"changed-weights")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            provenance.verify_manifest(self.adapter, self.base)

    def test_changed_config_rejected(self):
        (self.base / "config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            provenance.verify_manifest(self.adapter, self.base)

    def test_legacy_adapter_warns(self):
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(provenance.verify_manifest(self.root / "legacy", self.base))

    def test_missing_weights_rejected(self):
        (self.base / "model.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "weight files"):
            provenance.fingerprint_base(self.base)

    def test_full_model_has_no_manifest(self):
        output = self.root / "full"
        provenance.save_manifest(SimpleNamespace(), output)
        self.assertFalse(output.exists())

    def test_saved_artifact_inventory(self):
        (self.adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        provenance.save_manifest(SimpleNamespace(_xvla_base_manifest=self.manifest), self.adapter)
        manifest = json.loads((self.adapter / provenance.MANIFEST).read_text(encoding="utf-8"))
        self.assertIn("adapter_config.json", manifest["artifacts"])
        self.assertNotIn(provenance.MANIFEST, manifest["artifacts"])


if __name__ == "__main__":
    unittest.main()
