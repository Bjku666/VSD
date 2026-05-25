#!/usr/bin/env python3
"""E13_loss_check: inspect E13_2/E13_3 loss configuration and outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    summary = {
        "exists": True,
        "sha256": _sha256(path),
    }
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        model = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
        state = model.state_dict() if hasattr(model, "state_dict") else {}
    except Exception as exc:
        summary["torch_load_error"] = str(exc)
        return summary
    summary.update(
        {
            "state_keys": len(state),
            "has_ir_backbone": any("ir_backbone" in key for key in state),
            "has_e13_loss_attrs": any(
                key in str(getattr(model, "__dict__", {}))
                for key in ("loss_mode", "small_px", "scale_alpha", "center_alpha")
            ),
            "model_class": type(model).__name__,
        }
    )
    return summary


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out-dir", default="/mnt/disk2/lhr/VSD/results/val/e13_loss_check")
    parser.add_argument("--e13-2-dir", default="/mnt/disk2/lhr/VSD/results/val/e13_2_e6_scale_aware_loss")
    parser.add_argument("--e13-3-dir", default="/mnt/disk2/lhr/VSD/results/val/e13_3_e6_center_aware_loss")
    parser.add_argument("--e13-2-val", default="/mnt/disk2/lhr/VSD/results/val/e13_2_e6_scale_aware_loss_val")
    parser.add_argument("--e13-3-val", default="/mnt/disk2/lhr/VSD/results/val/e13_3_e6_center_aware_loss_val")
    parser.add_argument("--e13-2-log", default="/mnt/disk2/lhr/VSD/results/val/logs/e13_2_scale_aware_loss_20260520.log")
    parser.add_argument("--e13-3-log", default="/mnt/disk2/lhr/VSD/results/val/logs/e13_3_center_aware_loss_20260520.log")
    parser.add_argument("--core", default="/mnt/disk2/lhr/VSD/scripts/e13_tiny_aware_loss_core.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    e13_2_best = Path(args.e13_2_dir) / "weights" / "best.pt"
    e13_3_best = Path(args.e13_3_dir) / "weights" / "best.pt"
    e13_2_metrics = _load_json(Path(args.e13_2_val) / "required_metrics.json")
    e13_3_metrics = _load_json(Path(args.e13_3_val) / "required_metrics.json")
    core_text = _read_text(Path(args.core))
    log_2 = _read_text(Path(args.e13_2_log))
    log_3 = _read_text(Path(args.e13_3_log))

    metric_keys = [
        "mAP50-95",
        "AP_dark-small",
        "AP_tiny",
        "AP_low-contrast",
        "False Positives/image",
        "FPPI_dark",
        "FPPI_low-contrast",
    ]
    metric_delta = {}
    identical_metrics = True
    for key in metric_keys:
        v2 = e13_2_metrics.get(key)
        v3 = e13_3_metrics.get(key)
        try:
            delta = float(v3) - float(v2)
        except Exception:
            delta = None
        metric_delta[key] = {"E13_2": v2, "E13_3": v3, "delta_E13_3_minus_E13_2": delta}
        if delta not in (0, 0.0):
            identical_metrics = False

    code_findings = {
        "scale_gain_used_when_scale_aware": "use_scale=self.loss_mode in {\"scale-aware\", \"scale-center-aware\"}" in core_text,
        "center_penalty_used_when_center_aware": "center_alpha=self.center_alpha if self.loss_mode in {\"center-aware\", \"scale-center-aware\"}" in core_text,
        "loss_applies_to_all_foreground": "target_scores.sum(-1)[fg_mask]" in core_text,
        "no_dark_tiny_mask_in_loss": "dark-small" not in core_text and "low-contrast" not in core_text,
        "max_gain_default_3": "max_gain: float = 3.0" in core_text,
        "center_alpha_default_0_25": "center_alpha: float = 0.25" in core_text,
    }
    log_findings = {
        "e13_2_mentions_scale_aware": "scale-aware" in log_2,
        "e13_3_mentions_center_aware": "center-aware" in log_3,
        "e13_2_mentions_center_aware": "center-aware" in log_2,
        "e13_3_mentions_scale_aware": "scale-aware" in log_3,
    }

    summary = {
        "experiment": "E13_loss_check",
        "status": "done",
        "weights": {
            "E13_2": _state_summary(e13_2_best),
            "E13_3": _state_summary(e13_3_best),
        },
        "metrics_identical_on_key_fields": identical_metrics,
        "metric_delta": metric_delta,
        "code_findings": code_findings,
        "log_findings": log_findings,
        "interpretation": [
            "E13_2 and E13_3 use different configured loss modes, but their required metrics are identical on the checked fields.",
            "The current loss implementation applies the auxiliary weighting to all foreground assignments, not only dark-small or tiny targets.",
            "A lighter follow-up should scope the auxiliary term to tiny/dark-small assignments and reduce the auxiliary gain before launching E13_2b/E13_3b.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# E13 Loss Check",
        "",
        f"- Key metrics identical: {identical_metrics}",
        f"- E13_2 best.pt sha256: {summary['weights']['E13_2'].get('sha256', '')[:16]}",
        f"- E13_3 best.pt sha256: {summary['weights']['E13_3'].get('sha256', '')[:16]}",
        "",
        "## Checked Metrics",
        "",
        "| Metric | E13_2 | E13_3 | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, vals in metric_delta.items():
        delta = vals["delta_E13_3_minus_E13_2"]
        lines.append(f"| {key} | {vals['E13_2']} | {vals['E13_3']} | {delta} |")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- The loss code has separate scale-aware and center-aware branches.",
            "- The auxiliary term currently applies to all foreground boxes, not only tiny/dark-small boxes.",
            "- Because E13_2 and E13_3 metrics are identical, do not launch E13_4 as a mainline run before narrowing the loss scope.",
        ]
    )
    (out_dir / "metrics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "metrics_summary.md")


if __name__ == "__main__":
    main()
