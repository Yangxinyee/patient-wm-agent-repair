The separate action encoder offers a way to remove dose sensitivity without changing how measurements enter the model. Before doing that, I’ll check both sensors and several future horizons. Near-zero estimates across those checks would strengthen the null-effect hypothesis; consistent delayed or sensor-specific responses would refute it.

```python
import time
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

print("NUMERIC VOCAB:", json.dumps(VOCAB["numeric"], indent=2)[:6500])
print("NUMERIC NOTE:", VOCAB.get("numeric_note"))
feature_matrix = np.column_stack([np.asarray(v) for v in features.values()]).astype(np.float32)
analysis_hours = panel.index.get_level_values("hour").to_numpy()
analysis_stays = panel.index.get_level_values("stay_id").to_numpy()
analysis_fold = analysis_stays % 2
valid_time = (analysis_hours >= 0) & (analysis_hours <= 23)
null_checks = []
dose_residuals = {}
check_params = dict(max_iter=180, learning_rate=0.10, max_leaf_nodes=31,
                    min_samples_leaf=100, l2_regularization=5.0,
                    early_stopping=True, random_state=2026)
t0 = time.time()
with threadpool_limits(limits=8):
    for drug, outcomes in [("insulin", ["glucose_finger", "glucose_lab"]),
                           ("norepinephrine", ["map_arterial", "map_cuff"])]:
        observed_dose = panel[drug].to_numpy()
        dose_prediction = np.full(len(panel), np.nan)
        for fold in (0, 1):
            tr = valid_time & (analysis_fold != fold)
            te = valid_time & (analysis_fold == fold)
            reg = HistGradientBoostingRegressor(**check_params)
            reg.fit(feature_matrix[tr], observed_dose[tr])
            dose_prediction[te] = reg.predict(feature_matrix[te])
        dose_residuals[drug] = observed_dose - dose_prediction
        for outcome in outcomes:
            for horizon in (1, 4, 8):
                future = panel.groupby(level="stay_id")[outcome].shift(-horizon).to_numpy()
                current = panel[outcome].to_numpy()
                use = valid_time & (analysis_hours + horizon <= 24)
                use &= np.isfinite(current) & np.isfinite(future)
                positions = np.flatnonzero(use)
                target = future[use] - current[use]
                target_prediction = np.zeros(len(positions))
                folds = analysis_fold[use]
                for fold in (0, 1):
                    tr = folds != fold
                    te = folds == fold
                    reg = HistGradientBoostingRegressor(**check_params)
                    reg.fit(feature_matrix[positions[tr]], target[tr])
                    target_prediction[te] = reg.predict(feature_matrix[positions[te]])
                residual_dose = dose_residuals[drug][use]
                residual_y = target - target_prediction
                denominator = residual_dose @ residual_dose
                beta = (residual_dose @ residual_y) / denominator
                unique_stays, groups = np.unique(analysis_stays[use], return_inverse=True)
                cluster_scores = np.bincount(
                    groups, weights=residual_dose * (residual_y - beta * residual_dose))
                se = np.sqrt(cluster_scores @ cluster_scores) / denominator
                row = dict(drug=drug, outcome=outcome, horizon=horizon,
                           beta=float(beta), cluster_se=float(se), n=int(use.sum()))
                null_checks.append(row)
                print(row)
print("Elapsed seconds:", round(time.time() - t0, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 8: independent sensors and delayed-response checks\n")
    f.write(json.dumps(null_checks, indent=2) + "\n")
```
