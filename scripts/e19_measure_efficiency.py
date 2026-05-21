#!/usr/bin/env python3
"""E19 efficiency measurement demo CLI.

This is a numbered experiment demo entrypoint. It records the intended command
surface for the experiment plan but does not implement the experiment logic yet.
Use --dry-run to print the parsed command safely. Running without --dry-run exits
non-zero so unfinished experiments are not launched by accident.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dry-run", action="store_true", help="Print the planned invocation and exit successfully.")
    parser.add_argument("--out", default=None, help="Optional output file for demo metadata.")
    parser.add_argument("--out-dir", default=None, help="Optional output directory for demo metadata.")
    args, unknown = parser.parse_known_args()
    args.extra = unknown
    return args


def main() -> None:
    args = parse_args()
    payload = {
        "script": Path(__file__).name,
        "status": "demo_only",
        "dry_run": bool(args.dry_run),
        "extra_args": args.extra,
        "message": "This numbered experiment entrypoint is a safe demo placeholder; implement the model/evaluator before running for real.",
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)

    target = args.out
    if target is None and args.out_dir:
        target = str(Path(args.out_dir) / f"{Path(__file__).stem}_demo.json")
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")

    if not args.dry_run:
        raise SystemExit("demo-only script: rerun with --dry-run, or replace this placeholder with the implemented experiment.")


if __name__ == "__main__":
    main()
