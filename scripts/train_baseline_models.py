#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxoffice.db import connect
from boxoffice.train import apply_default_filters, train_two_baselines
from boxoffice.transform import fetch_training_frame


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline multiplier + intl/dom models")
    p.add_argument("--db", default="data/processed/boxoffice.sqlite")
    p.add_argument("--model-dir", default="data/models")
    p.add_argument("--no-filters", action="store_true", help="Disable default outlier filters")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    conn = connect(args.db)
    df = fetch_training_frame(conn)
    filtered = apply_default_filters(df) if not args.no_filters else df
    metrics = train_two_baselines(df, model_dir=args.model_dir, apply_filters=not args.no_filters)
    print(
        json.dumps(
            {
                "rows_before_filters": int(len(df)),
                "rows_after_filters": int(len(filtered)),
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print(f"Saved models to: {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
