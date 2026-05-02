#!/usr/bin/env python
"""
Simple zone-pair average baseline.

Uses (pickup_zone, dropoff_zone) → average duration.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"

    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p.name}. Run `python data/download_data.py` first."
            )

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)

    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    print("\nBuilding zone-pair averages...")

    # Global fallback
    global_avg = train["duration_seconds"].mean()

    # Route-wise average
    route_avg = (
        train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .mean()
        .to_dict()
    )

    print(f"  total routes: {len(route_avg):,}")

    print("\nPredicting on dev set...")
    t0 = time.time()

    # Prediction
    def predict(row):
        key = (row["pickup_zone"], row["dropoff_zone"])
        return route_avg.get(key, global_avg)

    preds = dev.apply(predict, axis=1)

    print(f"  prediction done in {time.time() - t0:.0f}s")

    # MAE calculation
    y_dev = dev["duration_seconds"]
    mae = float(np.mean(np.abs(preds - y_dev)))

    print(f"\nZone-pair Dev MAE: {mae:.1f} seconds")


if __name__ == "__main__":
    main()