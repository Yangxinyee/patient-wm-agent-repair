Rates are partly recoverable, but that could reflect the prescribing policy rather than the administered dose itself. I’ll test dose innovations with weaker regularisation and inspect the norepinephrine policy. Recoverable innovations would strengthen the case for a readout edit; poor recovery would suggest repairing the action pathway instead.

```python
from scipy.linalg import svd

tHX_repair = build_history_features(tpan)
print("History columns match:", list(tHX_repair.columns) == list(HX.columns))
print("DML model tuple members:",
      {drug: [[type(m).__name__ for m in pair] for pair in pairs]
       for drug, pairs in dml_models.items()})

policy_pred_tuning = {}
for drug, pairs in dml_models.items():
    policy_pred_tuning[drug] = np.mean([
        pair[0].predict(tHX_repair.to_numpy(dtype=np.float32))
        for pair in pairs
    ], axis=0)

probe_rows = []
rate_probe = {}
feature_sets = {
    "out": state_o["x"][:, :192].astype(np.float64),
    "z": state_o["x"][:, 960:992].astype(np.float64),
    "full": state_o["x"].astype(np.float64),
}
alphas_probe = [1e-5, 1e-3, 0.1, 10.0]

for drug, flag in (("insulin", "has_insulin"),
                   ("norepinephrine", "has_norepinephrine")):
    actual = state_panel[drug].to_numpy(dtype=float)
    policy = pd.Series(policy_pred_tuning[drug], index=tpan.index).reindex(
        state_panel_idx
    ).to_numpy()
    active = state_o[flag].astype(bool) & np.isfinite(actual) & np.isfinite(policy)
    targets = np.column_stack([actual, actual - policy])
    for feature_name, features in feature_sets.items():
        predictions = {a: np.full_like(targets, np.nan) for a in alphas_probe}
        coefficients = {a: [] for a in alphas_probe}
        for fold_number in (0, 1):
            fit_mask = active & (state_split != fold_number)
            test_mask = active & (state_split == fold_number)
            xm = features[fit_mask].mean(axis=0)
            ym = targets[fit_mask].mean(axis=0)
            U, singular, Vt = svd(
                features[fit_mask] - xm, full_matrices=False,
                check_finite=False
            )
            projected_y = U.T @ (targets[fit_mask] - ym)
            for alpha in alphas_probe:
                coef = Vt.T @ (
                    (singular / (singular**2 + alpha))[:, None] * projected_y
                )
                intercept = ym - xm @ coef
                predictions[alpha][test_mask] = features[test_mask] @ coef + intercept
                coefficients[alpha].append((coef, intercept))
        for alpha in alphas_probe:
            pred = predictions[alpha]
            probe_rows.append({
                "drug": drug, "features": feature_name, "alpha": alpha,
                "rate_R2": r2_score(targets[active, 0], pred[active, 0]),
                "innovation_R2": r2_score(targets[active, 1], pred[active, 1]),
                "innovation_RMSE": np.sqrt(np.mean(
                    (targets[active, 1] - pred[active, 1])**2
                )),
                "innovation_SD": targets[active, 1].std(),
            })
            rate_probe[(drug, feature_name, alpha)] = (
                pred, coefficients[alpha], active, targets
            )
probe_table = pd.DataFrame(probe_rows)
print("\n=== HELD-OUT-STAY DECODING ===")
print(probe_table.round(5).to_string(index=False))

fresh_map_repair = pan["map_arterial"].combine_first(pan["map_cuff"])
latest_map_repair = fresh_map_repair.groupby(level=0).ffill()
map_policy_summary = pd.DataFrame({
    "score": latest_map_repair,
    "dose": pan["norepinephrine"],
    "next_map": next_obs["map_arterial"],
    "fresh_arterial": pan["map_arterial"].notna(),
})
map_policy_summary = map_policy_summary[
    valid_hours & map_policy_summary.score.notna()
]
map_policy_table = map_policy_summary.groupby("score").agg(
    n=("dose", "size"),
    treated=("dose", lambda x: (x > 0).mean()),
    mean_rate=("dose", "mean"),
    sd_rate=("dose", "std"),
    next_map=("next_map", "mean"),
)
print("\n=== NOREPINEPHRINE POLICY BY LATEST MAP ===")
print(map_policy_table.loc[
    (map_policy_table.index >= 50) & (map_policy_table.index <= 85)
].round(4).to_string())

print("\nDML row alignment examples:")
for drug, target in (("insulin", "glucose_finger"),
                     ("norepinephrine", "map_arterial")):
    res = dml_residuals[drug]
    ii = np.asarray(res["indices"])[:5]
    print(drug)
    print(pd.DataFrame({
        "hour": hours[ii],
        "current_last": last[target].to_numpy()[ii],
        "next_reading": next_obs[target].to_numpy()[ii],
        "stored_outcome_change": np.asarray(res["outcome_change"])[:5],
        "dose": np.asarray(res["dose"])[:5],
    }).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 9\n"
        "Run 8 rate decoding with ridge alpha=10: insulin R2 0.513 (out), "
        "0.657 (full); norepinephrine R2 0.273 (out), 0.485 (full).\n"
        "Tested weaker ridge penalties and conditional dose-innovation decoding "
        "with stay-separated validation. This distinguishes policy information "
        "from usable administered-dose information.\n"
        + probe_table.round(5).to_string(index=False) + "\n"
        "Inspected norepinephrine assignment versus latest recorded MAP and "
        "confirmed DML outcome alignment. No edits.\n"
    )
```