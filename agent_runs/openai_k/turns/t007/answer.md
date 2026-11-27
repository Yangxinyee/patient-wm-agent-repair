Recorded histories explain about 87% of insulin-rate variation and 94% of norepinephrine-rate variation, but meaningful residual variation remains. I’ll use stay-held-out models to estimate how that residual variation relates to next-hour readings, checking sensors and treatment subgroups separately. Stable estimates would support a causal target; sensor-dependent or subgroup-dependent estimates would call for stronger identification checks.

```python
run_count += 1

def residual_cluster_fit(rt, ry, groups):
    Z = np.column_stack([np.ones(len(ry)), rt])
    gram = Z.T @ Z
    b = np.linalg.solve(gram, Z.T @ ry)
    err = ry - Z @ b
    _, gi = np.unique(groups, return_inverse=True)
    scores = np.zeros((gi.max()+1, Z.shape[1]))
    np.add.at(scores, gi, Z * err[:, None])
    inv = np.linalg.inv(gram)
    se = np.sqrt(np.diag(inv @ (scores.T @ scores) @ inv))
    return b[1:], se[1:]

dml_results = {}
outcome_residuals = {}
outcome_predictions = {}
print("=== CROSS-FITTED NEXT-HOUR ESTIMATES ===")
for code in measure_codes:
    yy = next_values[code].to_numpy()[causal_rows]
    mask = np.isfinite(yy)
    pred = np.full(len(yy), np.nan)
    for k in (0, 1):
        tr = mask & (fold != k)
        te = mask & (fold == k)
        model = HistGradientBoostingRegressor(
            max_iter=220, learning_rate=.07, max_leaf_nodes=31,
            min_samples_leaf=80, l2_regularization=5,
            early_stopping=False, random_state=31+k
        )
        model.fit(Xc[tr], yy[tr])
        pred[te] = model.predict(Xc[te])
    ry = yy - pred
    outcome_residuals[code] = ry
    outcome_predictions[code] = pred
    b, se = residual_cluster_fit(dose_resid[mask], ry[mask], gc[mask])
    dml_results[code] = {"beta": b, "se": se, "n": int(mask.sum())}
    print(code, "n", mask.sum(), "slopes", np.round(b, 4).tolist(),
          "cluster SE", np.round(se, 4).tolist(),
          "outcome residual SD", round(np.nanstd(ry), 4))
    ownj = 0 if code.startswith("glucose") else 1
    subdivisions = {
        "drug_positive": Tc[:, ownj] > 0,
        "current_sensor_observed": P[code].notna().to_numpy()[causal_rows],
        "both_current_sensors": (
            P[measure_codes[:2] if ownj == 0 else measure_codes[2:]]
            .notna().all(axis=1).to_numpy()[causal_rows]
        ),
        "hours_0_to_11": hc < 12,
        "hours_12_to_23": hc >= 12,
    }
    for label, ss in subdivisions.items():
        mm = mask & ss
        bb, ee = residual_cluster_fit(dose_resid[mm], ry[mm], gc[mm])
        print(" ", label, "n", mm.sum(), "own-drug slope",
              round(bb[ownj], 4), "SE", round(ee[ownj], 4))

print("\n=== POSITIVE-RATE POLICY EXAMPLES ===")
for drug, sensor, target_cur, target_prev in [
    ("insulin", "glucose_finger", 170., 1.5),
    ("insulin", "glucose_finger", 200., 2.),
    ("norepinephrine", "map_arterial", 65., .3),
    ("norepinephrine", "map_arterial", 60., .5),
]:
    mm = valid_time & (P[sensor].to_numpy() == target_cur) & (
        np.abs(previous_rate[drug].to_numpy() - target_prev) < 1e-8)
    s = P.loc[mm, drug]
    print(drug, target_cur, target_prev, "n", len(s),
          "top rates", s.value_counts(normalize=True).head(8).round(3).to_dict())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 5: cross-fitted next-hour effects\n"
        "- Dose-policy out-of-stay R2: insulin 0.8660, NE 0.9351; residual SD 0.3678 and 0.07559 respectively.\n"
        "- Used two-fold, stay-held-out nonlinear outcome models and clustered residual-on-residual regressions.\n"
        "- Checked sensor, positive-dose, and early/late subgroups; estimates still require confounding and dynamics checks.\n"
        f"- DML results: {dml_results!r}\n"
        f"- Python runs: {run_count}; forward passes: {forward_count}.\n"
    )
```
