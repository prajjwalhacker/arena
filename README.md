## Baseline Analysis

- Ran the provided baseline XGBoost model on the dev dataset
- Observed Dev MAE ≈ 355 seconds

## Zone-Pair Baseline

After analyzing the baseline model, I implemented a simple aggregation-based approach.

Approach:
- Group trips by (pickup_zone, dropoff_zone)
- Compute average duration for each route
- Use this as the prediction

Results:
- Baseline (XGBoost): ~355 seconds MAE
- Zone-pair average: ~300 seconds MAE

Insights:
- A simple aggregation outperforms the baseline ML model
- This indicates that route-level historical patterns are the strongest signal
- The baseline model fails to capture this directly and instead approximates using independent features

Conclusion:
- This problem behaves more like a lookup/aggregation problem than a pure ML problem
- Feature engineering is more impactful than model complexity