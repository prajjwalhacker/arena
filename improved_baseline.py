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
    "month",
    "passenger_count",
    "route_avg",
    "route_hour_avg",
]


def engineer_features(df: pd.DataFrame, route_avg, route_hour_avg, global_avg) -> pd.DataFrame:
    ts = pd.to_datetime(df["requested_at"])

    df = df.copy()
    df["hour"] = ts.dt.hour.astype("int8")
    df["dow"] = ts.dt.dayofweek.astype("int8")
    df["month"] = ts.dt.month.astype("int8")

    # Route avg
    df["route_avg"] = df.apply(
        lambda r: route_avg.get(
            (r["pickup_zone"], r["dropoff_zone"]),
            global_avg
        ),
        axis=1
    )

    # Route + hour avg
    df["route_hour_avg"] = df.apply(
        lambda r: route_hour_avg.get(
            (r["pickup_zone"], r["dropoff_zone"], r["hour"]),
            route_avg.get((r["pickup_zone"], r["dropoff_zone"]), global_avg)
        ),
        axis=1
    )

    return df[FEATURES]


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"

    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run `python data/download_data.py` first.")

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)

    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    print("\nBuilding aggregation features...")

    # Global avg
    global_avg = train["duration_seconds"].mean()

    # Route avg
    route_avg = (
        train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .mean()
        .to_dict()
    )

    # Route + hour avg
    train["hour"] = pd.to_datetime(train["requested_at"]).dt.hour

    route_hour_avg = (
        train.groupby(["pickup_zone", "dropoff_zone", "hour"])["duration_seconds"]
        .mean()
        .to_dict()
    )

    print("\nEngineering features...")
    X_train = engineer_features(train, route_avg, route_hour_avg, global_avg)
    y_train = train["duration_seconds"].to_numpy()

    X_dev = engineer_features(dev, route_avg, route_hour_avg, global_avg)
    y_dev = dev["duration_seconds"].to_numpy()

    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    t0 = time.time()
    model.fit(X_train, y_train, verbose=False)
    print(f"  trained in {time.time() - t0:.0f}s")

    preds = model.predict(X_dev)
    mae = float(np.mean(np.abs(preds - y_dev)))

    print(f"\nImproved Dev MAE: {mae:.1f} seconds")

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