#!/usr/bin/env python
"""Export a local XVLA adapter as a verified standalone deployment checkpoint."""

import argparse
import copy
import json
from pathlib import Path


def validate_paths(base: Path, adapter: Path, output: Path) -> tuple[Path, Path, Path]:
    base, adapter = base.resolve(strict=True), adapter.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise ValueError(f"Output must not already exist: {output}")
    for source in (base, adapter):
        if source in output.parents or output in source.parents:
            raise ValueError("Output must not overlap a source directory")
    for root, names in (
        (base, ("config.json", "model.safetensors")),
        (adapter, ("config.json", "adapter_config.json", "adapter_model.safetensors",
                   "policy_preprocessor.json", "policy_postprocessor.json", "xvla_base_manifest.json")),
    ):
        for name in names:
            if not (root / name).is_file():
                raise ValueError(f"Required file missing: {root / name}")
    return base, adapter, output


def export(args):
    import torch
    from peft import PeftConfig, PeftModel
    from safetensors.torch import load_file

    from lerobot.common.xvla_provenance import MANIFEST, _digest, verify_manifest
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

    base, adapter, output = validate_paths(args.base_model, args.adapter, args.output_dir)
    manifest = verify_manifest(adapter, base)
    if manifest is None:
        raise ValueError("A verified base manifest is required for export")
    # Check the saved adapter bundle before trusting its configs or extra module weights.
    for relative, expected in manifest.get("artifacts", {}).items():
        path = (adapter / relative).resolve()
        if adapter not in path.parents:
            raise ValueError(f"Unsafe artifact path: {relative}")
        if not path.is_file() or path.stat().st_size != expected["size"] or _digest(path) != expected["sha256"]:
            raise ValueError(f"Adapter artifact identity mismatch: {relative}")

    config = PreTrainedConfig.from_pretrained(adapter, local_files_only=True)
    if config.type != "xvla":
        raise ValueError("Only XVLA adapters are supported")
    config.device = args.device
    if args.dtype:
        config.dtype = args.dtype
    config.use_peft = False
    config.pretrained_path = str(base)
    peft_config = PeftConfig.from_pretrained(adapter, local_files_only=True)
    if str(peft_config.peft_type).split(".")[-1].upper() != "LORA":
        raise ValueError("Only LoRA adapters can be exported by this tool")
    model = XVLAPolicy.from_pretrained(base, config=config, local_files_only=True)
    wrapped = PeftModel.from_pretrained(model, adapter, config=peft_config, is_trainable=False)
    wrapped.to(args.device).eval()
    processors = make_pre_post_processors(
        policy_cfg=config, pretrained_path=adapter,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    batch = load_file(str(args.validation_batch), device=args.device)
    domain_key = config.domain_feature_key or "domain_id"
    if domain_key not in batch or not torch.all(batch[domain_key] == config.domain_id):
        raise ValueError(f"Validation batch must contain {domain_key}={config.domain_id}")

    def predict(policy):
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        policy.reset()
        with torch.inference_mode():
            value = policy.predict_action_chunk(copy.deepcopy(batch)).detach().float().cpu()
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite validation actions")
        return value

    before = predict(wrapped)
    extra_states = {
        name: {key: value.detach().cpu().clone() for key, value in module.modules_to_save["default"].state_dict().items()}
        for name, module in wrapped.get_base_model().named_modules()
        if hasattr(module, "modules_to_save") and "default" in module.modules_to_save
    }
    for target in peft_config.modules_to_save or []:
        if not any(name == target or name.endswith("." + target) for name in extra_states):
            raise ValueError(f"Expected extra trained module was not loaded: {target}")
    # PEFT unwraps modules_to_save as well as folding the low-rank updates.
    merged = wrapped.merge_and_unload(safe_merge=True)
    if any("lora_" in name or "modules_to_save" in name for name in merged.state_dict()):
        raise ValueError("PEFT wrapper parameters remain after merge")
    for name, expected in extra_states.items():
        actual = merged.get_submodule(name).state_dict()
        if actual.keys() != expected.keys():
            raise ValueError(f"Extra trained module keys changed during merge: {name}")
        for key, value in expected.items():
            torch.testing.assert_close(actual[key].cpu(), value, atol=0, rtol=0)
    after = predict(merged)
    torch.testing.assert_close(after, before, atol=args.atol, rtol=args.rtol)
    merged.config.use_peft = False
    merged.config.pretrained_path = None

    output.mkdir(parents=True, exist_ok=False)
    try:
        merged.save_pretrained(output)
        for processor in processors:
            processor.save_pretrained(output)
        # Pipelines save their tokenizer locally; the policy fallback uses this same resource.
        if not (output / "tokenizer").is_dir():
            raise ValueError("Processor export did not include tokenizer resources")
        merged.config.tokenizer_name = str(output / "tokenizer")
        merged.config.save_pretrained(output)
        loaded = XVLAPolicy.from_pretrained(output, local_files_only=True).to(args.device).eval()
        reloaded = predict(loaded)
        torch.testing.assert_close(reloaded, after, atol=args.atol, rtol=args.rtol)
        # Also verify pipeline resources can be reloaded from the new artifact.
        make_pre_post_processors(
            policy_cfg=loaded.config, pretrained_path=output,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        report = {
            "base": manifest["base"], "adapter": str(adapter), "seed": args.seed,
            "dtype": merged.config.dtype, "atol": args.atol, "rtol": args.rtol,
            "validation_batch_sha256": _digest(args.validation_batch),
            "merge_max_abs_error": (before - after).abs().max().item(),
            "reload_max_abs_error": (after - reloaded).abs().max().item(),
            "validated": True,
            "extra_modules_verified": list(extra_states),
        }
        (output / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception:
        (output / "EXPORT_FAILED.txt").write_text(
            "Export validation failed. Do not deploy this directory.\n", encoding="utf-8"
        )
        raise
    print(f"Verified standalone XVLA checkpoint: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True, help="pretrained_model directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must not exist")
    parser.add_argument("--validation-batch", type=Path, required=True,
                        help="Preprocessed observation tensors saved as safetensors")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.01)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("Tolerances must be nonnegative")
    export(args)


if __name__ == "__main__":
    main()
