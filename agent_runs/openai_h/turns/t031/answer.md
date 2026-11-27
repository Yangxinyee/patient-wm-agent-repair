The immediate insulin estimate remains near zero, but the post-hour-12 three-hour signal is counter-evidence that needs checking. Its outcome was a model-input feature, so I’ll first restore physical units and test that subgroup on the previously unused stays. A reproducible response would challenge the zero-effect edit; weak replication would favor retaining it.

```python
import numpy as np
import pandas as pd
import json

# Recover physical readings directly from the training records.
record_candidates = [
    (k, v) for k, v in list(globals().items())
    if isinstance(v, pd.DataFrame)
    and {"stay_id", "hour", "event", "value"}.issubset(v.columns)
    and len(v) >= len(Hrich)
]
if record_candidates:
    physical_record_name, physical_records = max(record_candidates, key=lambda kv: len(kv[1]))
else:
    physical_record_name = "/data/train.csv"
    physical_records = pd.read_csv(
        "/data/train.csv", usecols=["stay_id", "hour", "event", "value"])
assert physical_records["stay_id"].nunique() == 30000

physical_finger_rows = physical_records.loc[
    physical_records["event"].eq("glucose_finger"),
    ["stay_id", "hour", "value"],
].copy()
assert np.allclose(physical_finger_rows["hour"], np.round(physical_finger_rows["hour"]))
assert not physical_finger_rows.duplicated(["stay_id", "hour"]).any()

physical_finger = physical_finger_rows.set_index(["stay_id", "hour"])["value"].reindex(Hrich.index)
unit_mask = (
    np.isfinite(physical_finger.to_numpy())
    & np.isfinite(insulin_rep_current)
    & (insulin_rep_observed > 0.5)
)
unit_design = np.column_stack([
    np.ones(int(unit_mask.sum())),
    insulin_rep_current[unit_mask],
])
unit_intercept, unit_gain = np.linalg.lstsq(
    unit_design, physical_finger.to_numpy()[unit_mask], rcond=None)[0]
unit_errors = (
    physical_finger.to_numpy()[unit_mask]
    - unit_intercept - unit_gain * insulin_rep_current[unit_mask]
)
print("Physical-record source:", physical_record_name)
print("Physical glucose = intercept + gain * feature:",
      float(unit_intercept), float(unit_gain))
print("Maximum unit-mapping error:", float(np.max(np.abs(unit_errors))))
assert unit_gain > 0 and np.max(np.abs(unit_errors)) < 0.01

insulin_physical_results = {}
for horizon in (1, 3, 6):
    residual_y = unit_gain * (
        insulin_rep_targets[horizon] - insulin_rep_predictions[horizon])
    subsets = {
        "all_stays": insulin_rep_masks[horizon],
        "hours_12_on": insulin_rep_masks[horizon] & (insulin_rep_hour >= 12),
        "hours_12_on_independent_stays": (
            insulin_rep_masks[horizon]
            & (insulin_rep_hour >= 12)
            & (insulin_rep_fold != 0)
        ),
    }
    insulin_physical_results[str(horizon)] = {}
    for label, mask in subsets.items():
        insulin_physical_results[str(horizon)][label] = {
            "linear": audit_cluster_regression(
                insulin_rep_basis[mask, :1], residual_y[mask], insulin_rep_ids[mask]),
            "quadratic": audit_cluster_regression(
                insulin_rep_basis[mask], residual_y[mask], insulin_rep_ids[mask]),
        }
print("\n=== INSULIN RESPONSE IN mg/dL PER RATE UNIT ===")
print(json.dumps(insulin_physical_results, indent=2))

# Prepare a different diagnostic: next-hour glucose conditional on measured
# history, testing current and six past dose magnitudes jointly.
insulin_lag_rates = np.column_stack([
    insulin_rep_d if lag == 0 else
    pd.Series(insulin_rep_d).groupby(insulin_rep_ids, sort=False).shift(lag).fillna(0).to_numpy()
    for lag in range(7)
])
insulin_lag_presence = (insulin_lag_rates > 0).astype(np.float32)
insulin_lag_feature_names = [
    c for c in Hrich.columns if not c.startswith("insulin_lag")
]
insulin_lag_X = np.column_stack([
    Hrich[insulin_lag_feature_names].to_numpy(dtype=np.float32),
    insulin_lag_presence,
])
insulin_lag_basis = np.column_stack([
    insulin_lag_rates,
    insulin_lag_rates ** 2,
])
physical_next_finger = physical_finger.groupby(level="stay_id", sort=False).shift(-1).to_numpy()
insulin_lag_target = physical_next_finger - physical_finger.to_numpy()
insulin_lag_valid = (
    np.isfinite(insulin_lag_target)
    & (insulin_rep_hour >= 0)
    & (insulin_rep_hour <= 23)
    & insulin_lag_presence.any(axis=1)
)
print("\nNext-hour lag-dose diagnostic rows:", int(insulin_lag_valid.sum()),
      "features:", insulin_lag_X.shape[1],
      "dose terms:", insulin_lag_basis.shape[1])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 22 / forward passes 7: physical-unit and delayed-signal audit\n"
            "- Run 21 outcomes were scaled predictor features; verified an exact affine mapping to physical training glucose readings.\n"
            "- Physical mapping: intercept=" + str(float(unit_intercept)) +
            ", gain=" + str(float(unit_gain)) + ".\n"
            "- Immediate response remains near zero, but post-hour-12 three-hour response is counter-evidence and is not dismissed.\n"
            "- Physical-unit and independent-stay results: " + json.dumps(insulin_physical_results) + "\n"
            "- Prepared a joint current/past-dose next-hour diagnostic to distinguish delayed-dose effects from local-projection policy associations.\n"
            "- No weight changes.\n")
```