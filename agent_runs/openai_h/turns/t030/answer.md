The replication failed because a fitting subset was empty; it produced no new effect estimate and changed no weights. I’ll check the time units, rebuild the subsets from the panel’s timestamp index, and verify their sizes before fitting. Small replicated effects would support the edit; a stable nonzero immediate or delayed response would challenge it.

```python
import time
import numpy as np
import pandas as pd
import json
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

print("Feature-hour range:", float(Hrich["hour"].min()), float(Hrich["hour"].max()))
print("Index names:", Hrich.index.names)
print("Previous subset sizes:", {h: int(m.sum()) for h, m in insulin_rep_masks.items()})

assert isinstance(Hrich.index, pd.MultiIndex)
time_levels = [n for n in Hrich.index.names if n != "stay_id"]
assert len(time_levels) == 1, "Need an unambiguous raw timestamp level."
insulin_rep_hour = Hrich.index.get_level_values(time_levels[0]).to_numpy(dtype=float)
print("Raw-hour range:", float(insulin_rep_hour.min()), float(insulin_rep_hour.max()))
assert np.isfinite(insulin_rep_hour).all()
assert np.allclose(insulin_rep_hour, np.round(insulin_rep_hour))
assert insulin_rep_hour.max() >= 24

insulin_rep_panel = pd.DataFrame({
    "stay_id": insulin_rep_ids,
    "hour": insulin_rep_hour,
    "glucose": insulin_rep_current,
    "observed": insulin_rep_observed,
})
insulin_rep_group = insulin_rep_panel.groupby("stay_id", sort=False)
assert np.allclose(insulin_rep_group["hour"].diff().dropna(), 1.0)

insulin_rep_base = (
    (insulin_rep_d > 0)
    & (insulin_rep_observed > 0.5)
    & np.isfinite(insulin_rep_current)
    & (insulin_rep_hour >= 0)
    & (insulin_rep_hour <= 23)
)
insulin_rep_targets, insulin_rep_masks = {}, {}
for horizon in (1, 3, 6):
    future = insulin_rep_group["glucose"].shift(-horizon).to_numpy()
    future_observed = insulin_rep_group["observed"].shift(-horizon).to_numpy()
    future_hour = insulin_rep_group["hour"].shift(-horizon).to_numpy()
    insulin_rep_targets[horizon] = future - insulin_rep_current
    insulin_rep_masks[horizon] = (
        insulin_rep_base
        & (future_observed > 0.5)
        & np.isfinite(future)
        & np.isclose(future_hour, insulin_rep_hour + horizon)
        & (future_hour <= 24)
    )
    counts = [
        int((insulin_rep_masks[horizon] & (insulin_rep_fold == f)).sum())
        for f in range(5)
    ]
    print("Horizon", horizon, "held-out rows by fold:", counts)
    assert min(counts) > 1000

insulin_rep_predictions = {
    "dose": np.full(len(Hrich), np.nan),
    "dose_squared": np.full(len(Hrich), np.nan),
    **{h: np.full(len(Hrich), np.nan) for h in (1, 3, 6)},
}
insulin_rep_params = dict(audit_positive_params)
t0 = time.time()

with threadpool_limits(limits=8):
    for fold in range(5):
        fit_base = insulin_rep_base & (insulin_rep_fold != fold)
        test_base = insulin_rep_base & (insulin_rep_fold == fold)
        for label, target in (
            ("dose", insulin_rep_d),
            ("dose_squared", insulin_rep_d ** 2),
        ):
            estimator = HistGradientBoostingRegressor(**insulin_rep_params)
            estimator.fit(Xrich[fit_base], target[fit_base])
            insulin_rep_predictions[label][test_base] = estimator.predict(Xrich[test_base])

        for horizon in (1, 3, 6):
            fit_mask = insulin_rep_masks[horizon] & (insulin_rep_fold != fold)
            test_mask = insulin_rep_masks[horizon] & (insulin_rep_fold == fold)
            estimator = HistGradientBoostingRegressor(**insulin_rep_params)
            estimator.fit(Xrich[fit_mask], insulin_rep_targets[horizon][fit_mask])
            insulin_rep_predictions[horizon][test_mask] = estimator.predict(Xrich[test_mask])

        mask = insulin_rep_masks[1] & (insulin_rep_fold == fold)
        residual_d = insulin_rep_d[mask] - insulin_rep_predictions["dose"][mask]
        residual_y = insulin_rep_targets[1][mask] - insulin_rep_predictions[1][mask]
        print("Fold", fold, json.dumps(audit_cluster_regression(
            residual_d[:, None], residual_y, insulin_rep_ids[mask])),
            "elapsed_seconds", round(time.time() - t0, 1))

insulin_rep_basis = np.column_stack([
    insulin_rep_d - insulin_rep_predictions["dose"],
    insulin_rep_d ** 2 - insulin_rep_predictions["dose_squared"],
])
insulin_rep_results = {}
for horizon in (1, 3, 6):
    residual_y = insulin_rep_targets[horizon] - insulin_rep_predictions[horizon]
    subsets = {
        "all_training_stays": insulin_rep_masks[horizon],
        "hours_12_on": insulin_rep_masks[horizon] & (insulin_rep_hour >= 12),
        "excluding_original_held_out_stays": (
            insulin_rep_masks[horizon] & (insulin_rep_fold != 0)
        ),
    }
    insulin_rep_results[str(horizon)] = {}
    for label, mask in subsets.items():
        insulin_rep_results[str(horizon)][label] = {
            "linear": audit_cluster_regression(
                insulin_rep_basis[mask, :1], residual_y[mask], insulin_rep_ids[mask]),
            "linear_and_quadratic": audit_cluster_regression(
                insulin_rep_basis[mask], residual_y[mask], insulin_rep_ids[mask]),
        }

print("\n=== ALL-STAY INSULIN REPLICATION ===")
print(json.dumps(insulin_rep_results, indent=2))
print("Positive insulin rate quantiles:",
      np.quantile(insulin_rep_d[insulin_rep_base], [0.1, 0.5, 0.9]).tolist())
print("Elapsed seconds:", round(time.time() - t0, 1))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python runs 20–21 / forward passes 7: insulin replication\n"
            "- Run 20 failed on an empty subset before producing effect estimates; no weight changes.\n"
            "- Run 21 uses raw timestamp-index hours rather than the predictor's hour feature, with nonempty-fold and chronological-order assertions.\n"
            "- Five-fold stay-separated estimates on positive insulin infusions, using only pretreatment predictors.\n"
            "- Longer-horizon estimates include subsequent recorded treatment-policy responses and are not an oracle for the fixed-dose rollout.\n"
            "- Results: " + json.dumps(insulin_rep_results) + "\n"
            "- No weight changes; full tuning factual delta remains +0.000307.\n")
```