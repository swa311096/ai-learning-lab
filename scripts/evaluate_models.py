#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from boxoffice.db import connect
from boxoffice.train import FEATURE_COLUMNS, apply_default_filters, build_pipeline, load_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved baseline models against current training table")
    p.add_argument("--db", default="data/processed/boxoffice.sqlite")
    p.add_argument("--model-dir", default="data/models")
    p.add_argument("--top-errors", type=int, default=5)
    p.add_argument("--no-filters", action="store_true", help="Disable default outlier filters for evaluation")
    p.add_argument("--time-split", type=float, default=0.2, help="Fraction for chronological holdout (0-0.5)")
    p.add_argument("--output-json", default="data/processed/evaluation_report.json")
    return p.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    denom = np.maximum(np.abs(y_true), 1.0)
    err = y_pred - y_true
    return {
        "count": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err * err))),
        "mape_pct": float(np.mean(np.abs(err) / denom) * 100.0),
    }


def fetch_eval_frame(conn) -> pd.DataFrame:
    frame = pd.read_sql_query(
        """
        SELECT
            movie_id,
            title,
            release_date,
            opening_weekend_usd,
            day3_total_usd,
            day7_total_usd,
            fri_sat_change_pct,
            sat_sun_change_pct,
            sun_mon_change_pct,
            theaters_day1,
            theaters_day7,
            release_month,
            is_holiday_window,
            domestic_total_usd,
            international_total_usd,
            worldwide_total_usd,
            domestic_multiplier,
            intl_dom_ratio
        FROM training_examples
        """,
        conn,
    )
    frame["release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
    return frame


def time_split_report(frame: pd.DataFrame, target: str, test_fraction: float) -> Optional[Dict[str, object]]:
    target_frame = frame.dropna(subset=FEATURE_COLUMNS + [target, "release_date"]).copy()
    if len(target_frame) < 40:
        return None

    target_frame = target_frame.sort_values("release_date")
    split_idx = int(len(target_frame) * (1.0 - test_fraction))
    split_idx = max(20, min(split_idx, len(target_frame) - 20))

    train_df = target_frame.iloc[:split_idx]
    test_df = target_frame.iloc[split_idx:]
    if train_df.empty or test_df.empty:
        return None

    model = build_pipeline()
    model.fit(train_df[FEATURE_COLUMNS], train_df[target].astype(float))
    preds = model.predict(test_df[FEATURE_COLUMNS])

    return {
        "train_count": int(len(train_df)),
        "test_count": int(len(test_df)),
        "train_end_date": train_df["release_date"].max().date().isoformat(),
        "test_start_date": test_df["release_date"].min().date().isoformat(),
        "metrics": metrics(test_df[target].to_numpy(dtype=float), preds),
    }


def monthly_mae(frame: pd.DataFrame, actual_col: str, pred_col: str) -> list:
    df = frame.dropna(subset=["release_month", actual_col, pred_col]).copy()
    if df.empty:
        return []
    out = []
    grouped = df.groupby("release_month")
    for month, sub in sorted(grouped, key=lambda x: x[0]):
        mae = float(np.mean(np.abs(sub[pred_col].to_numpy() - sub[actual_col].to_numpy())))
        out.append({"month": int(month), "count": int(len(sub)), "mae": mae})
    return out


def save_report(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    conn = connect(args.db)

    raw_frame = fetch_eval_frame(conn)
    frame = raw_frame if args.no_filters else apply_default_filters(raw_frame)

    summary = {
        "rows": {
            "training_examples_before_filters": int(len(raw_frame)),
            "training_examples_after_filters": int(len(frame)),
            "with_domestic_multiplier": int(frame["domestic_multiplier"].notna().sum()),
            "with_intl_dom_ratio": int(frame["intl_dom_ratio"].notna().sum()),
        },
        "models": {},
        "backtest": {},
        "charts": {},
        "notes": [
            "in_sample metrics use saved models on current filtered table rows.",
            "backtest metrics retrain models using chronological split.",
        ],
    }

    model_multiplier_path = Path(args.model_dir) / "domestic_multiplier_rf.joblib"
    model_ratio_path = Path(args.model_dir) / "intl_dom_ratio_rf.joblib"

    if model_multiplier_path.exists():
        model_multiplier = load_model(str(model_multiplier_path))
        dom_df = frame.dropna(subset=FEATURE_COLUMNS + ["domestic_multiplier"]).copy()
        if not dom_df.empty:
            dom_pred = model_multiplier.predict(dom_df[FEATURE_COLUMNS])
            summary["models"]["domestic_multiplier"] = metrics(
                dom_df["domestic_multiplier"].to_numpy(dtype=float),
                dom_pred,
            )

            dom_df["pred_domestic_multiplier"] = dom_pred
            dom_df["pred_domestic_total"] = dom_df["opening_weekend_usd"] * dom_df["pred_domestic_multiplier"]
            if dom_df["domestic_total_usd"].notna().any():
                dom_total_df = dom_df.dropna(subset=["domestic_total_usd", "pred_domestic_total"])
                summary["models"]["domestic_total_from_multiplier"] = metrics(
                    dom_total_df["domestic_total_usd"].to_numpy(dtype=float),
                    dom_total_df["pred_domestic_total"].to_numpy(dtype=float),
                )
                summary["charts"]["domestic_total_monthly_mae"] = monthly_mae(
                    dom_total_df,
                    actual_col="domestic_total_usd",
                    pred_col="pred_domestic_total",
                )

            dom_df["abs_pct_error"] = (
                np.abs(dom_df["pred_domestic_multiplier"] - dom_df["domestic_multiplier"])
                / np.maximum(np.abs(dom_df["domestic_multiplier"]), 1.0)
            )
            worst = dom_df.sort_values("abs_pct_error", ascending=False).head(args.top_errors)
            summary["models"]["domestic_multiplier"]["worst_examples"] = [
                {
                    "title": r.title,
                    "release_date": r.release_date.date().isoformat() if pd.notna(r.release_date) else None,
                    "actual": float(r.domestic_multiplier),
                    "predicted": float(r.pred_domestic_multiplier),
                    "abs_pct_error": float(r.abs_pct_error * 100.0),
                    "opening_weekend_usd": float(r.opening_weekend_usd) if pd.notna(r.opening_weekend_usd) else None,
                    "domestic_total_usd": float(r.domestic_total_usd) if pd.notna(r.domestic_total_usd) else None,
                    "day3_total_usd": float(r.day3_total_usd) if pd.notna(r.day3_total_usd) else None,
                    "day7_total_usd": float(r.day7_total_usd) if pd.notna(r.day7_total_usd) else None,
                }
                for r in worst.itertuples(index=False)
            ]
            summary["models"]["domestic_multiplier"]["multiplier_formula"] = (
                "Multiplier = Total domestic ÷ Opening weekend (Fri+Sat+Sun)"
            )

    if model_ratio_path.exists():
        model_ratio = load_model(str(model_ratio_path))
        ratio_df = frame.dropna(subset=FEATURE_COLUMNS + ["intl_dom_ratio"]).copy()
        if not ratio_df.empty:
            ratio_pred = model_ratio.predict(ratio_df[FEATURE_COLUMNS])
            summary["models"]["intl_dom_ratio"] = metrics(
                ratio_df["intl_dom_ratio"].to_numpy(dtype=float),
                ratio_pred,
            )

    if model_multiplier_path.exists() and model_ratio_path.exists():
        both = frame.dropna(subset=FEATURE_COLUMNS + ["opening_weekend_usd", "domestic_total_usd", "international_total_usd"]).copy()
        if not both.empty:
            model_multiplier = load_model(str(model_multiplier_path))
            model_ratio = load_model(str(model_ratio_path))
            both["pred_multiplier"] = model_multiplier.predict(both[FEATURE_COLUMNS])
            both["pred_ratio"] = model_ratio.predict(both[FEATURE_COLUMNS])
            both["pred_domestic"] = both["opening_weekend_usd"] * both["pred_multiplier"]
            both["pred_international"] = both["pred_domestic"] * both["pred_ratio"]
            both["pred_worldwide"] = both["pred_domestic"] + both["pred_international"]

            summary["models"]["combined_totals"] = {
                "domestic": metrics(both["domestic_total_usd"].to_numpy(dtype=float), both["pred_domestic"].to_numpy(dtype=float)),
                "international": metrics(
                    both["international_total_usd"].to_numpy(dtype=float),
                    both["pred_international"].to_numpy(dtype=float),
                ),
            }
            ww = both.dropna(subset=["worldwide_total_usd"])
            if not ww.empty:
                summary["models"]["combined_totals"]["worldwide"] = metrics(
                    ww["worldwide_total_usd"].to_numpy(dtype=float),
                    ww["pred_worldwide"].to_numpy(dtype=float),
                )

    split = max(0.05, min(float(args.time_split), 0.5))
    dom_backtest = time_split_report(frame, "domestic_multiplier", split)
    ratio_backtest = time_split_report(frame, "intl_dom_ratio", split)
    if dom_backtest:
        summary["backtest"]["domestic_multiplier"] = dom_backtest
    if ratio_backtest:
        summary["backtest"]["intl_dom_ratio"] = ratio_backtest

    save_report(args.output_json, summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation report to: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
