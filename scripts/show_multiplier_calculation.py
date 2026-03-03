#!/usr/bin/env python3
"""
Show how actual domestic_multiplier is calculated for specific movies.
Dumps underlying numbers from movies_raw, daily_gross, and training_examples.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxoffice.db import connect
from boxoffice.train import FEATURE_COLUMNS, apply_default_filters, load_model
import pandas as pd
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description="Show multiplier calculation details")
    p.add_argument("--db", default="data/processed/boxoffice.sqlite")
    p.add_argument("--model-dir", default="data/models")
    p.add_argument("--top-errors", type=int, default=5, help="Show N worst prediction errors")
    p.add_argument("--output", "-o", help="Also write output to this file")
    args = p.parse_args()

    conn = connect(args.db)
    cur = conn.cursor()

    # Load model and get worst errors (same logic as evaluate_models)
    model_path = Path(args.model_dir) / "domestic_multiplier_rf.joblib"
    if not model_path.exists():
        print("Model not found. Run train_baseline_models.py first.")
        return 1

    frame = pd.read_sql_query(
        """
        SELECT movie_id, title, release_date, opening_weekend_usd, day3_total_usd, day7_total_usd,
               domestic_total_usd, domestic_multiplier, intl_dom_ratio,
               fri_sat_change_pct, sat_sun_change_pct, sun_mon_change_pct,
               theaters_day1, theaters_day7, release_month, is_holiday_window
        FROM training_examples
        """,
        conn,
    )
    frame = apply_default_filters(frame)
    frame = frame.dropna(subset=FEATURE_COLUMNS + ["domestic_multiplier"])

    model = load_model(str(model_path))
    preds = model.predict(frame[FEATURE_COLUMNS])
    frame = frame.copy()
    frame["pred_domestic_multiplier"] = preds
    frame["abs_pct_error"] = (
        np.abs(frame["pred_domestic_multiplier"] - frame["domestic_multiplier"])
        / np.maximum(np.abs(frame["domestic_multiplier"]), 1.0)
    ) * 100
    worst = frame.nlargest(args.top_errors, "abs_pct_error")

    rows = [
        (
            r.movie_id,
            r.title,
            r.release_date,
            r.opening_weekend_usd,
            r.day3_total_usd,
            r.day7_total_usd,
            r.domestic_total_usd,
            r.domestic_multiplier,
            r.pred_domestic_multiplier,
            r.abs_pct_error,
        )
        for r in worst.itertuples(index=False)
    ]

    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("=" * 80)
    out("DOMESTIC MULTIPLIER = TOTAL DOMESTIC / OPENING WEEKEND")
    out("=" * 80)
    out()
    out("Weekend = first Fri+Sat+Sun | Total = final domestic box office")
    out()
    out("=" * 80)
    out("WORST PREDICTION ERRORS")
    out("=" * 80)

    for movie_id, title, release_date, ow, day3, day7, domestic, mult, pred, err in rows:
        out()
        out(f"  {title}  ({release_date})")
        out(f"  Weekend (Fri-Sat-Sun):  ${ow:,.0f}" if ow else "  Weekend:  —")
        out(f"  Total domestic:         ${domestic:,.0f}" if domestic else "  Total domestic:  —")
        out(f"  Multiplier:             {mult:.2f}  (Total ÷ Weekend)")
        out(f"  Model predicted:        {pred:.2f}   Error: {err:.0f}%")
        out()

    conn.close()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(lines), encoding="utf-8")
        print(f"\nWrote to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
