The rollout confirms exactly zero shifts for all four interventions, with factual fit unchanged on the first batch. That validates the edit’s mechanics, not the null-effect conclusion. I’ll now test insulin’s policy thresholds and norepinephrine estimates in better-observed and previously treated groups. Reproducible nonzero responses would argue against keeping complete dose invariance.

```python
# Independent causal robustness checks; model edits remain unchanged.
t0 = time.time()
robust_checks = []
with threadpool_limits(limits=8):
    for drug, outcome, other in [
        ("insulin", "glucose_finger", "glucose_lab"),
        ("norepinephrine", "map_arterial", "map_cuff")
    ]:
        current = panel[outcome].to_numpy()
        future = panel[outcome + "_next"].to_numpy()
        lag_dose = panel[drug + "_lag"].to_numpy()
        common = valid_time & np.isfinite(current) & np.isfinite(future)
        strata = {
            "hours_12_23": analysis_hours >= 12,
            "previously_running": lag_dose > 0,
            "both_current_sensors": np.isfinite(panel[other].to_numpy()),
            "running_and_both_sensors": (lag_dose > 0) & np.isfinite(panel[other].to_numpy())
        }
        for label, stratum in strata.items():
            use = common & stratum
            pos = np.flatnonzero(use)
            folds = analysis_fold[use]
            d = panel[drug].to_numpy()[use]
            target = future[use] - current[use]
            dp, yp = np.zeros(len(pos)), np.zeros(len(pos))
            for fold in (0, 1):
                tr, te = folds != fold, folds == fold
                dr = HistGradientBoostingRegressor(**check_params)
                yr_model = HistGradientBoostingRegressor(**check_params)
                dr.fit(feature_matrix[pos[tr]], d[tr])
                yr_model.fit(feature_matrix[pos[tr]], target[tr])
                dp[te] = dr.predict(feature_matrix[pos[te]])
                yp[te] = yr_model.predict(feature_matrix[pos[te]])
            rd, ry = d - dp, target - yp
            den = rd @ rd
            beta = (rd @ ry) / den
            _, group = np.unique(analysis_stays[use], return_inverse=True)
            score = np.bincount(group, weights=rd * (ry - beta * rd))
            se = np.sqrt(score @ score) / den
            row = dict(drug=drug, stratum=label, beta=float(beta),
                       cluster_se=float(se), n=len(pos), dose_resid_sd=float(rd.std()))
            robust_checks.append(row)
            print(row)

# Local discontinuity checks at insulin policy boundaries seen in the records.
gf = panel.glucose_finger.to_numpy()
cuts = np.asarray([150, 180, 220, 260, 300])
nearest = cuts[np.argmin(np.abs(gf[:, None] - cuts[None, :]), axis=1)]
rd_checks = []
for bandwidth in (6, 10, 14):
    for horizon in (1, 4):
        future = panel.groupby(level=0).glucose_finger.shift(-horizon).to_numpy()
        use = valid_time & (analysis_hours + horizon <= 24)
        use &= np.isfinite(gf) & np.isfinite(future)
        use &= np.abs(gf - (nearest + .5)) <= bandwidth
        positions = np.flatnonzero(use)
        x = (gf[use] - (nearest[use] + .5)) / bandwidth
        z = (x > 0).astype(float)
        cut_fe = (nearest[use, None] == cuts[None, :]).astype(float)
        controls = [cut_fe, cut_fe * x[:, None], cut_fe * (x*x)[:, None],
                    cut_fe * (z*x)[:, None]]
        for feature in ("hour", "glucose_finger_lag1", "glucose_finger_baseline",
                        "glucose_lab_now", "glucose_lab_last_before",
                        "map_arterial_now", "map_cuff_now", "insulin_lag1"):
            v = np.asarray(features[feature])[use].astype(float)
            missing = ~np.isfinite(v)
            v[missing] = 0
            scale = max(float(v.std()), 1.0)
            controls.extend([(v / scale)[:, None], missing.astype(float)[:, None]])
        C = np.concatenate(controls, axis=1)
        d = panel.insulin.to_numpy()[use]
        y_target = future[use] - gf[use]
        residuals = np.column_stack([z, d, y_target])
        residuals -= C @ np.linalg.lstsq(C, residuals, rcond=1e-9)[0]
        zi, di, yi = residuals.T
        first_stage = (zi @ di) / (zi @ zi)
        denominator = zi @ di
        beta = (zi @ yi) / denominator
        _, group = np.unique(analysis_stays[use], return_inverse=True)
        score = np.bincount(group, weights=zi * (yi - beta * di))
        se = np.sqrt(score @ score) / abs(denominator)
        row = dict(bandwidth=bandwidth, horizon=horizon, dose_jump=float(first_stage),
                   beta=float(beta), cluster_se=float(se), n=len(positions))
        rd_checks.append(row)
        print("INSULIN THRESHOLD IV:", row)
print("Elapsed seconds:", round(time.time()-t0, 1))
print("\nPolicy near boundaries:")
for cutoff in cuts:
    sub = panel.loc[valid_time & np.isfinite(gf) & (np.abs(gf-cutoff) <= 2)]
    print(cutoff, sub.groupby("glucose_finger").insulin.agg(["count", "mean"]).round(3).to_dict())
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward passes 3–4: candidate A validation\n")
    f.write("First 100 tuning stays: original factual NLL 1.50467, edited 1.50465 (difference -0.00002). Dose-edit rollout confirms all four edited shifts exactly zero. Original shifts: insulin x2 +0.209, half -0.072; NE x2 -0.007, half +0.032.\n")
    f.write("\n## Run 10: causal robustness checks\n")
    f.write(json.dumps({"strata": robust_checks, "threshold_iv": rd_checks}, indent=2) + "\n")
```
