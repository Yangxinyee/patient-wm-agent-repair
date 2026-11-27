The audit found only pretreatment predictors, with no direct dose leakage. Delayed estimates are imprecise and do not show a consistent response. Since insulin is the main readout, I’ll now replicate its immediate and delayed estimates across all training stays using stay-separated folds. Consistently small estimates would strengthen the near-zero hypothesis; a reproducible response would argue for revising the edit.

```python
import time
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

t0 = time.time()

# Recover row-aligned stay IDs without using them as model predictors.
if isinstance(Hrich.index, pd.MultiIndex) and "stay_id" in Hrich.index.names:
    insulin_rep_ids = Hrich.index.get_level_values("stay_id").to_numpy()
else:
    metadata_candidates = [
        (k, v) for k, v in list(globals().items())
        if isinstance(v, pd.DataFrame)
        and len(v) == len(Hrich)
        and {"stay_id", "hour"}.issubset(v.columns)
        and np.allclose(v["hour"].to_numpy(), Hrich["hour"].to_numpy())
    ]
    assert metadata_candidates, "Need row-aligned stay metadata before proceeding."
    insulin_rep_ids = metadata_candidates[0][1]["stay_id"].to_numpy()

insulin_rep_hour = Hrich["hour"].to_numpy(dtype=float)
insulin_rep_fold = np.asarray(cross_ne_fold, dtype=int)
assert set(np.unique(insulin_rep_fold)) == set(range(5))
assert pd.Series(insulin_rep_fold).groupby(insulin_rep_ids).nunique().max() == 1

insulin_rep_d = np.asarray(D[:, DRUGS.index("insulin")], dtype=float)
insulin_rep_current = Hrich["glucose_finger_raw_current"].to_numpy(dtype=float)
insulin_rep_observed = Hrich["glucose_finger_observed"].to_numpy(dtype=float)

insulin_rep_panel = pd.DataFrame({
    "stay_id": insulin_rep_ids,
    "hour": insulin_rep_hour,
    "glucose": insulin_rep_current,
    "observed": insulin_rep_observed,
})
insulin_rep_group = insulin_rep_panel.groupby("stay_id", sort=False)
insulin_rep_base = (
    (insulin_rep_d > 0)
    & (insulin_rep_observed > 0.5)
    & np.isfinite(insulin_rep_current)
    & (insulin_rep_hour >= 0)
    & (insulin_rep_hour <= 23)
)

insulin_rep_targets = {}
insulin_rep_masks = {}
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

insulin_rep_predictions = {
    "dose": np.full(len(Hrich), np.nan),
    "dose_squared": np.full(len(Hrich), np.nan),
    **{h: np.full(len(Hrich), np.nan) for h in (1, 3, 6)},
}
insulin_rep_params = dict(audit_positive_params)
insulin_rep_results = {}

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
        print("Fold", fold,
              json.dumps(audit_cluster_regression(
                  residual_d[:, None], residual_y, insulin_rep_ids[mask])),
              "elapsed_seconds", round(time.time() - t0, 1))

insulin_rep_basis = np.column_stack([
    insulin_rep_d - insulin_rep_predictions["dose"],
    insulin_rep_d ** 2 - insulin_rep_predictions["dose_squared"],
])

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
print("Total elapsed seconds:", round(time.time() - t0, 1))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 20 / forward passes 7: all-stay insulin replication\n"
            "- Five-fold stay-separated nuisance predictions; positive insulin infusions and same-sensor glucose readings.\n"
            "- Predictors contain current pretreatment readings and past history, but no current dose or future readings.\n"
            "- Examined horizons 1, 3, and 6, including intervention-relevant hours and independent replication stays.\n"
            "- Longer-horizon local projections include subsequent recorded treatment-policy responses; they are diagnostics, not an oracle for the fixed-dose rollout.\n"
            "- Results: " + json.dumps(insulin_rep_results) + "\n"
            "- No weight changes.\n")
```