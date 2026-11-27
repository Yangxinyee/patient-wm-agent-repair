Insulin still shows no clear response, including curvature. Norepinephrine’s linear estimate is also near zero, but its quadratic test produced a nominal signal, so that deserves investigation. I’ll use stronger adjustment and same-sensor changes, then check separate held-out groups. A stable curve would challenge the null; specification-sensitive estimates would weaken it.

```python
import json
import time
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

necheck_current = panel["map_arterial"].to_numpy(dtype=np.float64)
necheck_next = panel.groupby(level="stay_id")["map_arterial"].shift(-1).to_numpy()
necheck_dose = D[:, 1].astype(np.float64)
necheck_valid = (
    np.asarray(eligible, dtype=bool)
    & (necheck_dose > 0)
    & np.isfinite(necheck_current)
    & np.isfinite(necheck_next)
)
necheck_fit = necheck_valid & ~np.asarray(held, dtype=bool)
necheck_test = necheck_valid & np.asarray(held, dtype=bool)

# Explicitly include readings known before the infusion; no future features.
necheck_current_sensors = panel[
    ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
].to_numpy(dtype=np.float32)
necheck_xfit = np.column_stack([
    Xrich[necheck_fit], necheck_current_sensors[necheck_fit]
])
necheck_xtest = np.column_stack([
    Xrich[necheck_test], necheck_current_sensors[necheck_test]
])
necheck_ids = stay_array[necheck_test]
necheck_parameters = dict(strong_params)
necheck_parameters.update(
    max_iter=500, max_leaf_nodes=63, min_samples_leaf=80,
    l2_regularization=5.0, early_stopping=False, random_state=81082,
)
necheck_models = {}
necheck_predictions = {}
necheck_start = time.time()

with threadpool_limits(limits=8):
    for necheck_label, necheck_target in [
        ("dose", necheck_dose),
        ("dose_squared", necheck_dose ** 2),
        ("map_change", necheck_next - necheck_current),
        ("map_next", necheck_next),
    ]:
        necheck_model = HistGradientBoostingRegressor(**necheck_parameters)
        necheck_model.fit(necheck_xfit, necheck_target[necheck_fit])
        necheck_models[necheck_label] = necheck_model
        necheck_predictions[necheck_label] = necheck_model.predict(necheck_xtest)

necheck_basis = np.column_stack([
    necheck_dose[necheck_test] - necheck_predictions["dose"],
    necheck_dose[necheck_test] ** 2 - necheck_predictions["dose_squared"],
])
necheck_unique_ids = np.unique(necheck_ids)
necheck_rng = np.random.default_rng(56117)
necheck_shuffled_ids = necheck_rng.permutation(necheck_unique_ids)
necheck_half = np.isin(
    necheck_ids, necheck_shuffled_ids[:len(necheck_shuffled_ids) // 2]
)
necheck_subsets = {
    "all": np.ones(len(necheck_ids), dtype=bool),
    "held_out_half_1": necheck_half,
    "held_out_half_2": ~necheck_half,
    "hours_12_to_23": hour_array[necheck_test] >= 12,
}
necheck_results = {}

for necheck_label, necheck_target in [
    ("map_change", necheck_next - necheck_current),
    ("map_next", necheck_next),
]:
    necheck_residual_y = (
        necheck_target[necheck_test] - necheck_predictions[necheck_label]
    )
    necheck_results[necheck_label] = {
        "prediction_rmse": float(np.sqrt(np.mean(necheck_residual_y ** 2))),
        "linear": audit_cluster_regression(
            necheck_basis[:, :1], necheck_residual_y, necheck_ids
        ),
        "quadratic_by_subset": {},
    }
    for necheck_subset_label, necheck_subset in necheck_subsets.items():
        necheck_results[necheck_label]["quadratic_by_subset"][necheck_subset_label] = (
            audit_cluster_regression(
                necheck_basis[necheck_subset],
                necheck_residual_y[necheck_subset],
                necheck_ids[necheck_subset],
            )
        )

print("Training rows:", int(necheck_fit.sum()),
      "held-out rows:", int(necheck_test.sum()))
print("Dose residual SD:", float(np.std(necheck_basis[:, 0])))
print(json.dumps(necheck_results, indent=2))
print("Elapsed seconds:", round(time.time() - necheck_start, 1))

with Path("/work/notes.md").open("a") as necheck_f:
    necheck_f.write(
        "\n## Python run 15 / forward passes 7: challenge nominal NE curvature\n"
        "- Positive-dose insulin slope -0.02898 (cluster SE 0.08416); "
        "quadratic joint p=0.881; no clear response in severity quartiles.\n"
        "- Positive-dose NE linear slope 0.11773 (SE 0.18047), but "
        "quadratic coefficients [-1.63545, 1.30584], joint p=0.0243. "
        "This is counter-evidence to the null and must not be ignored.\n"
        "- Refit richer nuisance models on same-sensor adjacent readings, "
        "explicit current readings, and both changes and levels; examine "
        "independent held-out stay halves and evaluation-relevant hours.\n"
        "- NE curvature audit: " + json.dumps(necheck_results) + "\n"
        "- No weight changes; current dose-invariant edit remains provisional.\n"
    )
```