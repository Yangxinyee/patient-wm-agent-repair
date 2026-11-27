The original model predicts modest responses: insulin doubling lowers glucose by 3.68, while norepinephrine doubling raises MAP by 0.47. These are predictions, not targets. I’ll now isolate dose variation unexplained by pre-dose history using cross-fitted flexible regressions. Agreement across sensors and data halves would strengthen the causal interpretation.

```python
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

threadpool_limits(limits=8)
torch.set_num_threads(8)
start = time.time()

# Condition on treatment presence to distinguish dose intensity from starting treatment.
E = np.column_stack([controls, presence]).astype(np.float32)
fold = stay_row % 2
dose_resid = np.zeros_like(dose)
outcome_resid = np.full_like(next_y, np.nan)
nuisance_scores = []

def nuisance_model():
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.075, max_leaf_nodes=31,
        max_depth=6, min_samples_leaf=80, l2_regularization=10,
        early_stopping=False, random_state=42
    )

for heldout in [0, 1]:
    fitting = fold != heldout
    testing = fold == heldout
    for j, drug in enumerate(DRUGS):
        fit_mask = fitting & (dose[:, j] > 0)
        test_mask = testing & (dose[:, j] > 0)
        reg = nuisance_model().fit(E[fit_mask], dose[fit_mask, j])
        pred = reg.predict(E[test_mask])
        dose_resid[test_mask, j] = dose[test_mask, j] - pred
        nuisance_scores.append({
            "fold": heldout, "target": drug,
            "n": int(test_mask.sum()),
            "residual_sd": float(np.std(dose_resid[test_mask, j])),
            "rmse": float(np.sqrt(np.mean((pred - dose[test_mask, j])**2)))
        })
    for j, sensor in enumerate(MEAS):
        fit_mask = fitting & np.isfinite(next_y[:, j])
        test_mask = testing & np.isfinite(next_y[:, j])
        reg = nuisance_model().fit(E[fit_mask], next_y[fit_mask, j])
        pred = reg.predict(E[test_mask])
        outcome_resid[test_mask, j] = next_y[test_mask, j] - pred
        nuisance_scores.append({
            "fold": heldout, "target": sensor,
            "n": int(test_mask.sum()),
            "residual_sd": float(np.std(outcome_resid[test_mask, j])),
            "rmse": float(np.sqrt(np.mean((pred - next_y[test_mask, j])**2)))
        })

print("=== CROSS-FITTED NUISANCE FITS ===")
print(pd.DataFrame(nuisance_scores).round(5).to_string(index=False))
dml_results = []
for j, sensor in enumerate(MEAS):
    for sample_label, extra_mask in [
        ("all next readings", np.ones(len(dose), dtype=bool)),
        ("current sensor observed", np.isfinite(current_y[:, j])),
    ]:
        mask = np.isfinite(outcome_resid[:, j]) & extra_mask
        X = np.column_stack([np.ones(mask.sum()), dose_resid[mask]])
        b, se, rsd = cluster_ols(X, outcome_resid[mask, j], stay_row[mask])
        dml_results.append({
            "sensor": sensor, "sample": sample_label, "n": int(mask.sum()),
            "insulin": b[1], "insulin_se": se[1],
            "norepinephrine": b[2], "norepinephrine_se": se[2],
        })
print("\n=== RESIDUALISED DOSE ESTIMATES ===")
print(pd.DataFrame(dml_results).round(5).to_string(index=False))
print("Elapsed seconds:", round(time.time() - start, 1))

print("\n=== EXAMPLE DRUG TRAJECTORIES ===")
for drug in DRUGS:
    candidate = train[train.event == drug].groupby("stay_id").size().idxmax()
    print("Stay", candidate, "drug", drug)
    print(panel.loc[candidate].iloc[:13].round(3).to_string())

pd.DataFrame(dml_results).to_csv("/work/dml_results.csv", index=False)
with open("/work/notes.md", "a") as f:
    f.write("\n## Baseline rollout and flexible residualisation\n")
    f.write("- Original tuning first 100: insulin x2 -3.68; half +2.018; "
            "NE x2 +0.465; half -0.871.\n")
    f.write("- Two-fold cross-fitting by stay; flexible history controls and treatment-presence indicators.\n")
    f.write(pd.DataFrame(dml_results).to_string(index=False) + "\n")
    f.write("- Residualised estimates still require overlap, placebo and stability checks.\n")
    f.write("- Python runs: 4. Forward passes: 1. No edits yet.\n")
```