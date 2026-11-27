The final batch also preserved factual fit, with NLL lower by 0.00002 nats. I’ll pool all 400 stays and test late-period response slopes across low- and high-policy-dose groups. A consistent nonzero response—especially when both sensors are observed—would challenge the zero-effect interpretation.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import time

# Pool all paired factual validations of candidate A.
paired_frames = []
for ep in sorted(Path("/work").glob("fwd_*/factual_edited.csv")):
    op = ep.with_name("factual_original.csv")
    if not op.exists():
        continue
    fo, fe = pd.read_csv(op), pd.read_csv(ep)
    assert len(fo) == len(fe)
    identifying = [c for c in ("stay_id", "hour", "event", "value")
                   if c in fo.columns and c in fe.columns]
    assert fo[identifying].equals(fe[identifying]), ep
    nll_col = next((c for c in fo.columns if c.lower() == "nll"), None)
    if nll_col is None:
        nll_col = next(c for c in fo.columns if "nll" in c.lower())
    pair = fo[identifying].copy()
    pair["original_nll"] = fo[nll_col].to_numpy()
    pair["edited_nll"] = fe[nll_col].to_numpy()
    pair["delta_nll"] = pair["edited_nll"] - pair["original_nll"]
    paired_frames.append(pair)

factual_pairs = pd.concat(paired_frames, ignore_index=True)
factual_delta = float(factual_pairs.delta_nll.mean())
scores = factual_pairs.assign(
    score=factual_pairs.delta_nll - factual_delta
).groupby("stay_id").score.sum()
G = len(scores)
factual_se = float(np.sqrt(G / (G - 1) * np.square(scores).sum())
                   / len(factual_pairs))
factual_summary = {
    "stays": G,
    "readings": len(factual_pairs),
    "original_nll": float(factual_pairs.original_nll.mean()),
    "edited_nll": float(factual_pairs.edited_nll.mean()),
    "delta": factual_delta,
    "stay_cluster_se": factual_se,
    "delta_95ci": [factual_delta - 1.96 * factual_se,
                   factual_delta + 1.96 * factual_se],
}
print("=== POOLED FACTUAL VALIDATION ===")
print(json.dumps(factual_summary, indent=2))
if "event" in factual_pairs:
    print(factual_pairs.groupby("event")[
        ["original_nll", "edited_nll", "delta_nll"]
    ].mean().to_string())

# Cross-fitted outcome changes, using pre-action history and the other drug.
# Groups depend on predicted policy dose, not the realized dose magnitude.
assert panel.index.equals(Xhistory.index)
assert len(panel) == len(analysis_stays) == len(analysis_hours)
fold_values = np.unique(analysis_fold)
print("\nCross-fit folds:", fold_values.tolist())
late_checks = []
started = time.time()
params = dict(check_params)
params["max_iter"] = min(params.get("max_iter", 120), 120)

def clustered_slope(r, y, ids):
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    r = r - r.mean()
    y = y - y.mean()
    denom = float(r @ r)
    if denom < 1e-10:
        return {"n": len(r), "unidentified": True}
    beta = float(r @ y / denom)
    score = pd.Series(r * (y - beta * r)).groupby(
        np.asarray(ids), sort=False
    ).sum().to_numpy()
    groups = len(score)
    se = float(np.sqrt(groups / max(groups - 1, 1)
                       * np.sum(score ** 2)) / denom)
    return {"n": len(r), "beta": beta, "cluster_se": se,
            "residual_rate_sd": float(r.std())}

with threadpool_limits(limits=8):
    for drug, outcome, companion, other in [
        ("insulin", "glucose_finger", "glucose_lab", "norepinephrine"),
        ("norepinephrine", "map_arterial", "map_cuff", "insulin"),
    ]:
        D = panel[drug].fillna(0).to_numpy(dtype=float)
        R = np.asarray(dose_residuals[drug], dtype=float)
        policy_dose = D - R
        now = panel[outcome].to_numpy(dtype=float)
        context = Xhistory.copy()
        context[other + "_current"] = panel[other].fillna(0)
        for horizon in (1, 4):
            future = panel[outcome].groupby(
                analysis_stays, sort=False
            ).shift(-horizon).to_numpy(dtype=float)
            eligible = (
                (analysis_hours >= 12)
                & (analysis_hours <= 24 - horizon)
                & np.isfinite(now) & np.isfinite(future)
                & np.isfinite(R) & np.isfinite(policy_dose)
            )
            idx = np.flatnonzero(eligible)
            XX = context.iloc[idx]
            target = future[idx] - now[idx]
            ff = np.asarray(analysis_fold)[idx]
            prediction = np.empty(len(idx))
            for fold in fold_values:
                test = ff == fold
                if not test.any():
                    continue
                reg = HistGradientBoostingRegressor(**params)
                reg.fit(XX.iloc[np.flatnonzero(~test)], target[~test])
                prediction[test] = reg.predict(
                    XX.iloc[np.flatnonzero(test)]
                )
            outcome_resid = target - prediction
            exposed = D[idx] > 0
            q25, q75 = np.quantile(policy_dose[idx][exposed], [0.25, 0.75])
            previous_rate = Xhistory[drug + "_lag1"].to_numpy()
            groups = {
                "all_with_current_reading": np.ones(len(idx), dtype=bool),
                "low_policy_dose": policy_dose[idx] <= q25,
                "high_policy_dose": policy_dose[idx] >= q75,
                "previously_running_both_sensors": (
                    (previous_rate[idx] > 0)
                    & panel[companion].iloc[idx].notna().to_numpy()
                ),
            }
            for label, select in groups.items():
                if select.sum() < 500:
                    continue
                result = clustered_slope(
                    R[idx][select], outcome_resid[select],
                    np.asarray(analysis_stays)[idx][select]
                )
                result.update(drug=drug, horizon=horizon, stratum=label)
                late_checks.append(result)

print("\n=== LATE-PERIOD POLICY-DOSE CHECKS ===")
print(pd.DataFrame(late_checks).round(5).to_string(index=False))
print("Elapsed seconds:", round(time.time() - started, 1))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward passes 6–7 and run 13\n"
        "Third tuning batch: NLL delta -0.00011; fourth: -0.00002. "
        "All 400 tuning stays now have paired factual validation.\n"
        "Pooled factual summary:\n" + json.dumps(factual_summary) + "\n"
        "Late-period, cross-fitted outcome-change slopes by predicted "
        "policy dose and sensor completeness (observational checks, "
        "not proof of identification):\n"
        + json.dumps(late_checks) + "\n"
    )
```