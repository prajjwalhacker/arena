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

## Feature Engineering Improvements

To improve the baseline model, I extended the feature set with route-level and time-aware signals.

### Approach
- Added route_avg: average duration for each (pickup_zone, dropoff_zone)
- Added route_hour_avg: average duration for each (pickup_zone, dropoff_zone, hour)
- Retained the original XGBoost model and training pipeline

### Results
- Baseline (XGBoost): ~355 seconds MAE
- Improved model: ~292 seconds MAE

### Observations
- Route-level aggregation introduces a strong signal and improves performance
- Time-of-day (hour) helps capture traffic variations
- However, the model does not fully leverage these features and still underperforms compared to pure aggregation approaches

### Insight
- This problem appears to be heavily driven by historical route patterns
- Simply adding features is not sufficient — the model must be guided to effectively use them