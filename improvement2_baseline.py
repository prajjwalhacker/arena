#!/usr/bin/env python

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

FEATURES = [
    "pickup_zone",
    "dropoff_zone",
    "hour",
    "dow",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])

    return pd.DataFrame({
        "pickup_zone": df["pickup_zone"].astype("int32"),
        "dropoff_zone": df["dropoff_zone"].astype("int32"),
        "hour": ts.dt.hour.astype("int8"),
        "dow": ts.dt.dayofweek.astype("int8"),
    })


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"

    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run download script first.")

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)

    print(f"train: {len(train):,}")
    print(f"dev:   {len(dev):,}")

    print("\nBuilding aggregations...")

    # Global fallback
    global_avg = train["duration_seconds"].mean()

    # Add hour
    train["hour"] = pd.to_datetime(train["requested_at"]).dt.hour
    dev["hour"] = pd.to_datetime(dev["requested_at"]).dt.hour

    # Route avg
    route_avg = (
        train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .mean()
        .to_dict()
    )

    # Route + hour avg (main signal)
    route_hour_avg = (
        train.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"]
        .mean()
        .to_dict()
    )

    print("Engineering features...")

    X_train = engineer_features(train)
    y_train = train["duration_seconds"].to_numpy()

    X_dev = engineer_features(dev)
    y_dev = dev["duration_seconds"].to_numpy()

    print("\nTraining XGBoost...")

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    t0 = time.time()
    model.fit(X_train, y_train, verbose=False)
    print(f"trained in {time.time() - t0:.0f}s")

    print("\nRunning hybrid prediction...")

    # ML predictions
    preds_ml = model.predict(X_dev)

    # Aggregation predictions
    def agg_predict(row):
        key1 = (row["pickup_zone"], row["dropoff_zone"], row["hour"])
        key2 = (row["pickup_zone"], row["dropoff_zone"])

        if key1 in route_hour_avg:
            return route_hour_avg[key1]
        if key2 in route_avg:
            return route_avg[key2]
        return global_avg

    preds_agg = dev.apply(agg_predict, axis=1).values

    # 🔥 Hybrid (key improvement)
    preds = 0.7 * preds_agg + 0.3 * preds_ml

    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"\nHybrid Dev MAE: {mae:.1f} seconds")

    # Save everything
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "route_avg": route_avg,
            "route_hour_avg": route_hour_avg,
            "global_avg": global_avg,
        }, f)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()