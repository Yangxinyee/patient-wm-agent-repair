The original rollout predicts small shifts: −1.58 and +0.91 for insulin, and +0.18 and −0.75 for norepinephrine. Their size and asymmetry do not establish whether they are wrong. I’ll now estimate responses to doses that were unexpected given the recorded history, with prediction models fitted on different stays. Stable estimates across histories and channels would support an effect; near-zero or unstable estimates would challenge it.

```python
python_runs += 1
forward_passes = 1
from sklearn.ensemble import HistGradientBoostingRegressor

print("=== ORIGINAL ROLLOUT FILE ===")
baseline_rollout = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print(baseline_rollout.columns.tolist())
print(baseline_rollout.head(4).to_string(index=False))

# All features precede the current infusion; no next-hour information is used.
Ccols, Cnames = [], []
for event in sensors:
    scale = 50.0 if "glucose" in event else 20.0
    center = np.nanmedian(P[event])
    for lag in (0, 1):
        Ccols.append((L[event][:, 1-lag:25-lag] - center) / scale)
        Cnames.append(event + "_last_lag" + str(lag))
        Ccols.append(Age[event][:, 1-lag:25-lag] / 10.0)
        Cnames.append(event + "_age_lag" + str(lag))
    obs = np.isfinite(P[event])
    cumulative_sum = np.cumsum(np.nan_to_num(P[event]), axis=1)
    cumulative_n = np.cumsum(obs, axis=1)
    prior_mean = cumulative_sum / np.maximum(cumulative_n, 1)
    prior_mean[cumulative_n == 0] = center
    Ccols.append((prior_mean[:, 1:25] - center) / scale)
    Cnames.append(event + "_past_mean")
    Ccols.append(np.broadcast_to((L[event][:, 1:2] - center) / scale, (ns, 24)))
    Cnames.append(event + "_initial")
    Ccols.append(obs[:, 1:25].astype(float))
    Cnames.append(event + "_observed_now")
Ccols.extend([tgrid / 24.0])
Cnames.append("hour")
for drug in drugs:
    scale = 1.0 if drug == "insulin" else 0.2
    for lag in (1, 2, 3):
        start = 1-lag
        a = np.zeros((ns, 24))
        if start >= 0:
            a[:] = R[drug][:, start:start+24]
        else:
            a[:, -start:] = R[drug][:, :24+start]
        Ccols.append(a / scale)
        Cnames.append(drug + "_lag" + str(lag))
C = np.stack(Ccols, axis=-1).astype(np.float32)

fold_by_stay = np.random.default_rng(941).integers(0, 2, size=ns)
causal_estimates = []
causal_residuals = {}

for event, drug in [
    ("glucose_finger", "insulin"), ("glucose_lab", "insulin"),
    ("map_arterial", "norepinephrine"), ("map_cuff", "norepinephrine")
]:
    y0, y1 = P[event][:, 1:25], P[event][:, 2:26]
    valid = np.isfinite(y0) & np.isfinite(y1)
    ii, jj = np.where(valid)
    Y = (y1 - y0)[valid]
    D = R[drug][:, 1:25][valid]
    other_drug = drugs[1] if drug == drugs[0] else drugs[0]
    X = np.column_stack([C[valid], R[other_drug][:, 1:25][valid]]).astype(np.float32)
    folds = fold_by_stay[ii]
    mD, mY = np.zeros_like(D), np.zeros_like(Y)
    for fold in (0, 1):
        fit, test = folds != fold, folds == fold
        params = dict(max_iter=100, max_leaf_nodes=15, learning_rate=.1,
                      min_samples_leaf=100, l2_regularization=10,
                      max_bins=63, early_stopping=False, random_state=81+fold)
        dose_model = HistGradientBoostingRegressor(**params).fit(X[fit], D[fit])
        outcome_model = HistGradientBoostingRegressor(**params).fit(X[fit], Y[fit])
        mD[test] = dose_model.predict(X[test])
        mY[test] = outcome_model.predict(X[test])
    rD, rY = D-mD, Y-mY
    causal_residuals[(event, drug)] = (ii, jj, D, rD, rY, mD, mY)
    for label, select in [
        ("all", np.ones(len(D), dtype=bool)),
        ("treated", D > 0),
        ("hours_6plus", jj >= 6)
    ]:
        rd, ry, si = rD[select], rY[select], ii[select]
        denom = rd @ rd
        beta = (rd @ ry) / denom
        score = np.bincount(si, weights=rd*(ry-beta*rd), minlength=ns)
        se = np.sqrt(score @ score) / denom
        causal_estimates.append(dict(
            event=event, drug=drug, subset=label, n=int(select.sum()),
            dose_residual_sd=float(rd.std()), beta=float(beta), cluster_se=float(se)
        ))
    print("Completed:", event)

print("\n=== CROSS-FITTED RESIDUAL DOSE ASSOCIATIONS ===")
print(pd.DataFrame(causal_estimates).round(5).to_string(index=False))
print("\n=== DRUG / MEASUREMENT OVERLAP BY STAY ===")
for drug in drugs:
    treated = (R[drug] > 0).any(axis=1)
    print(drug, {e: int((treated & np.isfinite(P[e]).any(axis=1)).sum()) for e in sensors})

with notes_path.open("a") as f:
    f.write(
        "\nRun 3 formatting failure fixed: use plain text, not optional tabulate.\n"
        + pd.DataFrame(association_results).to_string(index=False)
        + "\n## Forward 1: original dose rollout, 100 tuning stays\n"
        "Insulin x2=-1.575, half=+0.910; norepinephrine x2=+0.178, half=-0.752. "
        "Norepinephrine arterial readout reaches only 13 stays in this batch.\n"
        "\n## Run 4: cross-fitted dose-innovation diagnostics\n"
        "Flexible dose/outcome regressions use only pre-infusion records, "
        "initial measurements, prior means, reading ages, and previous doses. "
        "Folds are separated by stay. Cluster standard errors use stays. "
        "Residual associations still require checking unmeasured confounding "
        "and outcome-history measurement noise.\n"
        + pd.DataFrame(causal_estimates).to_string(index=False)
        + f"\nCounts: {python_runs} Python runs, {forward_passes} forward passes.\n"
    )
```
