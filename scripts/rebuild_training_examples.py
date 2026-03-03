#!/usr/bin/env python3
"""Rebuild training_examples from existing movies_raw and daily_gross (no scraping)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxoffice.db import connect
from boxoffice.transform import rebuild_training_examples


def main() -> int:
    p = argparse.ArgumentParser(description="Rebuild training examples from existing DB")
    p.add_argument("--db", default="data/processed/boxoffice.sqlite")
    p.add_argument("--all-studios", action="store_true", help="Include all studios; default is US only")
    args = p.parse_args()

    conn = connect(args.db)
    count = rebuild_training_examples(conn, as_of_day=7, us_only=not args.all_studios)
    conn.close()
    print(f"Training rows: {count} (us_only={not args.all_studios})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
