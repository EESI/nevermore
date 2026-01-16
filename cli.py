from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import NevermorePipeline, STAGE_ORDER


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the Nevermore pipeline with caching.")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("nevermore/configs/default.yaml"),
        help="Path to YAML config (default: nevermore/configs/default.yaml)",
    )
    ap.add_argument(
        "--up-to",
        default="report",
        choices=STAGE_ORDER,
        help="Run pipeline up to this stage (default: report)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    pipeline = NevermorePipeline(config_path=args.config)
    results = pipeline.run(up_to=args.up_to)
    for name, res in results.items():
        outputs = ", ".join(f"{k}={v}" for k, v in res.outputs.items()) if res.outputs else "no outputs"
        note = res.details.get("reason") if res.details else ""
        print(f"[{name}] signature={res.signature} outputs: {outputs} {note}")


if __name__ == "__main__":
    main()
