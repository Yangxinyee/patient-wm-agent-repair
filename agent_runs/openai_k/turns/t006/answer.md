The factual baseline is 1.63155 nats per reading on these 100 tuning stays. Before choosing an edit, I’ll test how well recorded histories explain dose selection. If unexplained dose variation remains, it may help identify effects; if histories almost determine the doses, ordinary adjusted regression could still be misleading.

```python
run_count += 1
forward_count = 2
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
threadpool_limits(8)

factual0 = pd.read_csv("/work/fwd_2/factual_original.csv")
rollout0 = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print("factual columns:", factual0.columns.tolist())
print(factual0.head(3).to_string(index=False))
print("rollout columns:", rollout0.columns.tolist())
print(rollout0.head(3).to_string(index=False))

obs_times = pd.DataFrame(
    np.where(P[measure_codes].notna(), hours[:, None], np.nan),
    index=P.index, columns=measure_codes
).groupby(level=0, sort=False).ffill()
ages = hours[:, None] - obs_times.to_numpy()
feature_parts = [
    hours[:, None].astype(float),
    last_values,
    P[measure_codes].notna().to_numpy().astype(float),
    ages,
    base_values,
]
feature_names = (
    ["hour"] + ["last_"+c for c in measure_codes]
    + ["observed_"+c for c in measure_codes]
    + ["age_"+c for c in measure_codes]
    + ["baseline_"+c for c in measure_codes]
)
for lag in (1, 2, 3):
    feature_parts.append(last.groupby(level=0, sort=False).shift(lag).to_numpy())
    feature_names += [f"last_lag{lag}_{c}" for c in measure_codes]
    feature_parts.append(rates.groupby(level=0, sort=False).shift(lag).to_numpy())
    feature_names += [f"rate_lag{lag}_{c}" for c in drug_codes]
cumr = rates.groupby(level=0, sort=False).cumsum().to_numpy() - rates.to_numpy()
feature_parts.append(cumr / np.maximum(1, hours[:, None]))
feature_names += ["past_mean_rate_"+c for c in drug_codes]
Xhistory = np.column_stack(feature_parts).astype(np.float32)
causal_rows = np.flatnonzero(valid_time)
Xc = Xhistory[causal_rows]
Tc = T_all[causal_rows]
gc = sid[causal_rows]
hc = hours[causal_rows]
fold = ((gc - ids[0]) % 2).astype(int)
dose_pred = np.empty_like(Tc)
dose_models = {}

start = time.time()
for j, drug in enumerate(drug_codes):
    for k in (0, 1):
        tr, te = fold != k, fold == k
        model = HistGradientBoostingRegressor(
            max_iter=160, learning_rate=.08, max_leaf_nodes=31,
            min_samples_leaf=100, l2_regularization=5,
            early_stopping=False, random_state=17+k
        )
        model.fit(Xc[tr], Tc[tr, j])
        dose_pred[te, j] = model.predict(Xc[te])
        dose_models[(drug, k)] = model

dose_resid = Tc - dose_pred
print("\n=== OUT-OF-STAY DOSE PREDICTION ===")
for j, drug in enumerate(drug_codes):
    r2 = 1 - np.mean(dose_resid[:, j]**2) / np.var(Tc[:, j])
    nz = Tc[:, j] > 0
    print(drug, "R2", round(r2, 6),
          "residual SD", round(np.std(dose_resid[:, j]), 6),
          "positive-rate residual SD", round(np.std(dose_resid[nz, j]), 6),
          "residual quantiles",
          np.round(np.quantile(dose_resid[:, j], [.01,.1,.5,.9,.99]), 5).tolist())

print("\n=== CONDITIONAL DOSE EXAMPLES ===")
for drug, sensor in [("insulin", "glucose_finger"),
                     ("norepinephrine", "map_arterial")]:
    tmp = pd.DataFrame({
        "current": P[sensor],
        "previous_rate": previous_rate[drug],
        "rate": P[drug]
    }).loc[valid_time].dropna()
    groups = tmp.groupby(["current", "previous_rate"], observed=True)
    largest = groups.size().nlargest(5)
    for key, n in largest.items():
        s = groups.get_group(key).rate
        print(drug, "current/previous", key, "n", n,
              "rates", s.value_counts(normalize=True).head(5).round(3).to_dict())

print("features", len(feature_names), "elapsed seconds", round(time.time()-start, 1))
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 4 / forwards 1–2: baseline and dose-policy models\n"
        "- Original tuning-100 rollout: insulin x2 -2.016; insulin half +1.051; NE x2 +0.309; NE half -0.070.\n"
        "- Original factual value NLL: 1.63155 over 4,751 readings.\n"
        "- Fit two-fold, stay-held-out flexible dose-policy models using only pre-dose histories.\n"
        "- These models are for identification diagnostics, not weight training.\n"
        f"- Python runs: {run_count}; forward passes: {forward_count}.\n"
    )
```
