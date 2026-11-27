The retained analysis contains pre-dose histories and cross-fitted dose residuals, which let us investigate effects beyond the next hour. That matters because repeated-dose rollouts depend on persistence as well as the immediate effect. I’ll estimate lagged effects while subtracting effects of later recorded doses. Stable lag patterns would support a rollout target; inconsistent patterns would weaken that interpretation.

```python
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
threadpool_limits(8)
torch.set_num_threads(8)

print("=== NUMERIC NORMALIZATION ===")
for spec in VOCAB["numeric"]:
    print(spec)
print("numeric_note:", VOCAB.get("numeric_note"))
print("\nPanel index:", panel.index.names, list(panel.index[:3]), list(panel.index[-3:]))
print("Hours:", np.unique(tt))
print("Observation fractions:", panel.notna().mean().round(3).to_dict())
print("Original rollout labels:", dose_base["edit"].unique().tolist())
print(dose_base.head(3).to_string(index=False))
print("Nuisance residual diagnostics:")
for drug in ("insulin", "norepinephrine"):
    rr = np.asarray(dose_residuals[drug])
    print(drug, "shape", rr.shape, "finite", np.isfinite(rr).sum(),
          "active residual SD", np.nanstd(rr[panel[drug].to_numpy() > 0]))

# Sequential g-estimation: at horizon h, remove estimated contributions
# of doses at t+1,...,t+h-1, then partial out only history known at t.
# These are observational estimates, not ground-truth oracle readouts.
irf_features = HC.to_numpy(dtype=np.float32)
irf_groups = panel.index.get_level_values(0).to_numpy()
irf_hour = panel.index.get_level_values(1).to_numpy(dtype=float)
irf_results = []
irf_coefficients = {}
irf_models = {}
irf_rng = np.random.default_rng(8162)
irf_start = time.time()

for sensor, drug in (("glucose_finger", "insulin"), ("map_arterial", "norepinephrine")):
    action = panel[drug].to_numpy(dtype=float)
    innovation = np.asarray(dose_residuals[drug], dtype=float)
    future_actions = {
        lag: panel.groupby(level=0)[drug].shift(-lag).to_numpy(dtype=float)
        for lag in range(1, 8)
    }
    coefficients = []
    for horizon in range(1, 9):
        target = panel.groupby(level=0)[sensor].shift(-horizon).to_numpy(dtype=float)
        for lag in range(1, horizon):
            target = target - coefficients[horizon-lag-1] * future_actions[lag]
        eligible_irf = (
            np.isfinite(target) & np.isfinite(innovation) & (action > 0)
            & (irf_hour >= 1) & (irf_hour + horizon <= 24)
        )
        prediction = np.full(len(panel), np.nan)
        fitted_pair = []
        for test_fold in (0, 1):
            fitting = np.flatnonzero(eligible_irf & (fold != test_fold))
            testing = np.flatnonzero(eligible_irf & (fold == test_fold))
            if len(fitting) > 80000:
                fitting = irf_rng.choice(fitting, 80000, replace=False)
            nuisance = HistGradientBoostingRegressor(
                max_iter=140, learning_rate=0.08, max_leaf_nodes=24,
                min_samples_leaf=80, l2_regularization=10,
                max_bins=127, early_stopping=False,
                random_state=8162 + horizon + test_fold
            )
            nuisance.fit(irf_features[fitting], target[fitting])
            prediction[testing] = nuisance.predict(irf_features[testing])
            fitted_pair.append(nuisance)
        idx = np.flatnonzero(eligible_irf & np.isfinite(prediction))
        rr = innovation[idx]
        residual = target[idx] - prediction[idx]
        denominator = np.dot(rr, rr)
        coefficient = np.dot(rr, residual) / denominator
        score = rr * (residual - coefficient * rr)
        cluster_scores = pd.Series(score).groupby(irf_groups[idx]).sum().to_numpy()
        se = np.sqrt(np.dot(cluster_scores, cluster_scores)) / denominator
        coefficients.append(float(coefficient))
        irf_models[(sensor, horizon)] = fitted_pair
        irf_results.append({
            "sensor": sensor, "horizon": horizon, "n": len(idx),
            "controlled_impulse": coefficient, "conditional_cluster_se": se
        })
    irf_coefficients[drug] = np.array(coefficients)

irf_table = pd.DataFrame(irf_results)
print("\n=== CONTROLLED LAGGED EFFECT ESTIMATES ===")
print(irf_table.round(4).to_string(index=False))
print("Elapsed seconds:", round(time.time() - irf_start, 1))

with open("/work/notes.md", "a") as f:
    f.write("\n## Lagged-effect investigation — Python 8; forward 2\n"
            "- Estimated controlled impulse responses using cross-fitted pre-dose history adjustment and sequential subtraction of subsequent dose effects.\n"
            "- Excluded hour 0. Standard errors condition on earlier estimated lag coefficients and therefore omit their propagated uncertainty.\n")
    f.write(irf_table.to_string(index=False) + "\n")
```
