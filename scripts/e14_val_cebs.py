#!/usr/bin/env python3
"""Validate E14 CEBS by reusing the E6 validation protocol."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

scripts_dir = os.environ.get("VSD_E6_SCRIPTS_DIR", "") or str(Path(__file__).resolve().parent)
if scripts_dir and scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import e6_val_feature_fusion_multiscale as e6val
from e14_cebs_core import E14CEBSTrainer


class _ConfiguredE14CEBSTrainer(E14CEBSTrainer):
    cebs_args: argparse.Namespace | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cebs_args
        if cfg is not None:
            self.set_cebs_config(
                cebs_alpha=cfg.cebs_alpha,
                dark_threshold=cfg.dark_threshold,
                low_contrast_threshold=cfg.low_contrast_threshold,
                contrast_kernel=cfg.contrast_kernel,
                suppression_temperature=cfg.suppression_temperature,
            )


def _parse_cebs_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--cebs-alpha", type=float, default=0.05)
    parser.add_argument("--dark-threshold", type=float, default=33.50320816040039)
    parser.add_argument("--low-contrast-threshold", type=float, default=0.08425217866897583)
    parser.add_argument("--contrast-kernel", type=int, default=7)
    parser.add_argument("--suppression-temperature", type=float, default=0.08)
    return parser.parse_known_args(argv)


def main() -> None:
    cebs_args, remaining = _parse_cebs_args(sys.argv[1:])
    _ConfiguredE14CEBSTrainer.cebs_args = cebs_args
    e6val.E6DetectionTrainer = _ConfiguredE14CEBSTrainer
    sys.argv = [sys.argv[0], *remaining]
    e6val.main()


if __name__ == "__main__":
    main()
