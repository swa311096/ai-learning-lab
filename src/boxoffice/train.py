from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "opening_weekend_usd",
    "day3_total_usd",
    "day7_total_usd",
    "fri_sat_change_pct",
    "sat_sun_change_pct",
    "sun_mon_change_pct",
    "theaters_day1",
    "theaters_day7",
    "release_month",
    "is_holiday_window",
]


def _build_pipeline() -> Pipeline:
    numeric_features = FEATURE_COLUMNS

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    return Pipeline(steps=[("pre", preprocessor), ("model", model)])


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100
    mae = mean_absolute_error(y_true, y_pred)
    return {"mae": float(mae), "mape_pct": float(mape)}


def train_two_baselines(df: pd.DataFrame, model_dir: str) -> Dict[str, Dict[str, float]]:
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    frame = df.copy()

    metrics: Dict[str, Dict[str, float]] = {}

    for target in ["domestic_multiplier", "intl_dom_ratio"]:
        target_frame = frame.dropna(subset=[target]).copy()
        if len(target_frame) < 20:
            raise ValueError(f"Not enough rows to train {target}; need at least 20 examples")

        X = target_frame[FEATURE_COLUMNS]
        y = target_frame[target].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )

        pipe = _build_pipeline()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics[target] = _evaluate(y_test.to_numpy(), preds)

        out_path = Path(model_dir) / f"{target}_rf.joblib"
        joblib.dump(pipe, out_path)

    return metrics


def load_model(model_path: str):
    return joblib.load(model_path)


def predict_totals(model_multiplier, model_ratio, features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["predicted_multiplier"] = model_multiplier.predict(out[FEATURE_COLUMNS])
    out["predicted_domestic_total"] = out["opening_weekend_usd"] * out["predicted_multiplier"]
    out["predicted_intl_dom_ratio"] = model_ratio.predict(out[FEATURE_COLUMNS])
    out["predicted_international_total"] = out["predicted_domestic_total"] * out["predicted_intl_dom_ratio"]
    out["predicted_worldwide_total"] = out["predicted_domestic_total"] + out["predicted_international_total"]
    return out
