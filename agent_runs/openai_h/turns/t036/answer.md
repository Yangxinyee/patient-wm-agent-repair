The broader norepinephrine analysis still permits a small effect, but its curvature is much weaker in the relevant post-hour-12 period and on independent stays. The factual-fit check remains comfortably within budget.

Before accepting dose invariance, I’ll test current and past norepinephrine doses jointly for a delayed response. A coherent, replicated lag pattern would refute the near-zero target; small, unstable contrasts would support retaining the edit.

```python
import numpy as np
import pandas as pd
import time
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ne_lag_start = time.time()
ne_lag_sid = panel.index.get_level_values(0).to_numpy()
ne_lag_hour = panel.index.get_level_values(1).to_numpy(dtype=float)
ne_lag_rate = panel["norepinephrine"].fillna(0).astype(float)
ne_lag_current_y = panel["map_arterial"].to_numpy(dtype=float)
ne_lag_next_y = (
    panel["map_arterial"].groupby(level=0, sort=False).shift(-1)
    .to_numpy(dtype=float)
)
ne_lag_y = ne_lag_next_y - ne_lag_current_y

# These are current/past doses only; never future doses or readings.
ne_lag_d = np.column_stack([
    ne_lag_rate.groupby(level=0, sort=False).shift(j).fillna(0).to_numpy()
    for j in range(7)
])
ne_lag_basis = np.column_stack([ne_lag_d, ne_lag_d ** 2])
ne_lag_valid = (
    (ne_lag_d[:, 0] > 0)
    & np.isfinite(ne_lag_current_y)
    & np.isfinite(ne_lag_next_y)
    & (ne_lag_hour >= 0)
    & (ne_lag_hour <= 23)
)

# Remove own-dose magnitude controls. Retain pretreatment measurements,
# other-drug histories, and own-dose presence history.
ne_lag_control_cols = [
    c for c in Hrich.columns if not c.startswith("norepinephrine")
]
ne_lag_X = Hrich[ne_lag_control_cols].copy()
for c in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    ne_lag_X["raw_current_" + c] = panel[c].to_numpy()
for j in range(7):
    ne_lag_X["ne_present_lag_" + str(j)] = (ne_lag_d[:, j] > 0).astype(float)

ne_lag_R = np.full(ne_lag_basis.shape, np.nan)
ne_lag_ry = np.full(len(panel), np.nan)
ne_lag_params = dict(
    max_iter=200, learning_rate=0.07, max_leaf_nodes=20,
    min_samples_leaf=150, l2_regularization=10.0,
    early_stopping=False, random_state=271
)

with threadpool_limits(limits=8):
    for fold in range(5):
        fit_mask = ne_lag_valid & (cross_ne_fold != fold)
        test_mask = ne_lag_valid & (cross_ne_fold == fold)
        assert fit_mask.any() and test_mask.any()
        assert not np.intersect1d(
            np.unique(ne_lag_sid[fit_mask]),
            np.unique(ne_lag_sid[test_mask])
        ).size
        Xfit = ne_lag_X.loc[fit_mask]
        Xtest = ne_lag_X.loc[test_mask]
        for j in range(14):
            model = HistGradientBoostingRegressor(**ne_lag_params)
            model.fit(Xfit, ne_lag_basis[fit_mask, j])
            ne_lag_R[test_mask, j] = (
                ne_lag_basis[test_mask, j] - model.predict(Xtest)
            )
        model = HistGradientBoostingRegressor(**ne_lag_params)
        model.fit(Xfit, ne_lag_y[fit_mask])
        ne_lag_ry[test_mask] = ne_lag_y[test_mask] - model.predict(Xtest)
        print("Completed NE lag fold", fold,
              "elapsed_seconds", round(time.time() - ne_lag_start, 1))

def lag_cluster_fit(R, y, sid, contrast_vectors):
    R = np.asarray(R, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.ones(len(y)), R])
    bread = np.linalg.pinv(X.T @ X)
    b = bread @ (X.T @ y)
    error = y - X @ b
    unique_sid, inverse = np.unique(sid, return_inverse=True)
    scores = np.zeros((len(unique_sid), X.shape[1]))
    np.add.at(scores, inverse, X * error[:, None])
    correction = (
        len(unique_sid) / (len(unique_sid) - 1)
        * (len(y) - 1) / (len(y) - X.shape[1])
    )
    covariance = correction * bread @ (scores.T @ scores) @ bread
    bc, vc = b[1:], covariance[1:, 1:]
    result = {
        "n": len(y), "stays": len(unique_sid),
        "linear_lags": bc[:7].tolist(),
        "linear_se": np.sqrt(np.maximum(0, np.diag(vc)[:7])).tolist(),
        "quadratic_lags": bc[7:].tolist(),
        "quadratic_se": np.sqrt(np.maximum(0, np.diag(vc)[7:])).tolist(),
        "joint_p": float(chi2.sf(
            bc @ np.linalg.pinv(vc) @ bc, len(bc)
        ))
    }
    for label, vector in contrast_vectors.items():
        result[label] = {
            "conditional_shift_mmHg": float(vector @ bc),
            "cluster_se": float(np.sqrt(max(0, vector @ vc @ vector)))
        }
    return result

ne_lag_results = {}
ne_lag_independent = ~np.isin(ne_lag_sid, cross_ne_original_held_stays)

for label, mask in {
    "all_stays": ne_lag_valid,
    "hours_12_to_23": ne_lag_valid & (ne_lag_hour >= 12),
    "independent_hours_12_to_23":
        ne_lag_valid & (ne_lag_hour >= 12) & ne_lag_independent
}.items():
    readout_mask = mask & (ne_lag_hour >= 12)
    # Contrasts change only infusions starting from hour 12.
    changed = (
        ne_lag_hour[readout_mask, None] - np.arange(7)[None, :] >= 12
    )
    dose = ne_lag_d[readout_mask] * changed
    sid = ne_lag_sid[readout_mask]
    _, inv = np.unique(sid, return_inverse=True)
    counts = np.bincount(inv)
    weights = 1.0 / counts[inv]
    weights /= weights.sum()
    contrasts = {}
    for factor, name in [(2.0, "x2"), (0.5, "half")]:
        delta_basis = np.column_stack([
            (factor - 1) * dose,
            (factor ** 2 - 1) * dose ** 2
        ])
        contrasts[name] = weights @ delta_basis
    ne_lag_results[label] = lag_cluster_fit(
        ne_lag_R[mask], ne_lag_ry[mask], ne_lag_sid[mask], contrasts
    )
    print(label, json.dumps(ne_lag_results[label]))

print("\n=== WEIGHT-INVARIANCE CHECK ===")
nw = load("embed.numeric.weight")
nw0 = load_original("embed.numeric.weight")
print("Dose/rate numeric-column maximum:",
      float(nw[:, [2, 4]].abs().max()))
print("Measurement-column change:",
      float((nw[:, :2] - nw0[:, :2]).abs().max()))
for name in ["embed.bins.dose.weight", "embed.bins.rate.weight"]:
    w, w0 = load(name), load_original(name)
    print(name, "positive-row spread:",
          float((w[1:] - w[1:2]).abs().max()),
          "absent-row change:", float((w[0] - w0[0]).abs().max()))
print("Next validation IDs:", list(EVAL_IDS)[300:400])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python run 27: joint NE lag challenge\n"
        "- Five-fold stay-separated residualization of current and six past "
        "positive-dose magnitudes and their squares; retained measurements "
        "and drug presence, removed own-dose magnitude nuisance controls.\n"
        "- Contrasts are conditional on measured histories, not true rollout effects.\n"
        "- Results: " + json.dumps(ne_lag_results) + "\n"
        "- No weight changes. Forward passes: 7.\n"
    )
print("Elapsed seconds:", round(time.time() - ne_lag_start, 1))
```
