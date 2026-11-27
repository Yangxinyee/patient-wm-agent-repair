The held-out estimates now broadly agree across sensors: about −2 glucose units per additional insulin unit/hour, and roughly +3 MAP units per additional norepinephrine rate unit. Cross-drug associations are small. I’ll next estimate how long these effects persist, accounting for clinicians’ later dose changes. Persistent effects would imply much larger rollout shifts than the original model predicts; rapid decay would argue against that conclusion.

```python
run_count += 1
max_horizon = 6
future_rate_residuals = {0: dose_resid.copy()}
dynamic_outcome_residuals = {}

start = time.time()
for q in range(1, max_horizon):
    future_t = rates.groupby(level=0, sort=False).shift(-q).to_numpy()[causal_rows]
    mask = np.isfinite(future_t).all(axis=1)
    predictions = np.full_like(future_t, np.nan)
    for j in range(2):
        for k in (0, 1):
            tr, te = mask & (fold != k), mask & (fold == k)
            model = HistGradientBoostingRegressor(
                max_iter=180, learning_rate=.08, max_leaf_nodes=31,
                min_samples_leaf=100, l2_regularization=5,
                early_stopping=False, random_state=61+k
            )
            model.fit(Xc[tr], future_t[tr, j])
            predictions[te, j] = model.predict(Xc[te])
    future_rate_residuals[q] = future_t - predictions

dynamic_moments = {}
dynamic_ir = {}
cluster_dynamic = {}
for code, j in [("glucose_lab", 0), ("map_arterial", 1)]:
    rhs, design = [], []
    cluster_y, cluster_a = [], []
    for h in range(1, max_horizon+1):
        yy = P[code].groupby(level=0, sort=False).shift(-h).to_numpy()[causal_rows]
        mask = np.isfinite(yy)
        if h == 1:
            ry = outcome_residuals[code].copy()
        else:
            predictions = np.full(len(yy), np.nan)
            for k in (0, 1):
                tr, te = mask & (fold != k), mask & (fold == k)
                model = HistGradientBoostingRegressor(
                    max_iter=220, learning_rate=.07, max_leaf_nodes=31,
                    min_samples_leaf=80, l2_regularization=5,
                    early_stopping=False, random_state=81+k
                )
                model.fit(Xc[tr], yy[tr])
                predictions[te] = model.predict(Xc[te])
            ry = yy - predictions
        dynamic_outcome_residuals[(code, h)] = ry
        mm = mask & (hc <= 24-max_horizon)
        instrument = dose_resid[mm, j]
        yprod = instrument * ry[mm]
        row = np.zeros(max_horizon)
        aprods = np.zeros((mm.sum(), max_horizon))
        for q in range(h):
            product = instrument * future_rate_residuals[q][mm, j]
            row[h-1-q] = product.sum()
            aprods[:, h-1-q] = product
        rhs.append(yprod.sum())
        design.append(row)
        gi = (gc[mm] - ids[0]).astype(int)
        cy = np.bincount(gi, weights=yprod, minlength=len(ids))
        ca = np.zeros((len(ids), max_horizon))
        np.add.at(ca, gi, aprods)
        cluster_y.append(cy)
        cluster_a.append(ca)
    D, R = np.array(design), np.array(rhs)
    ir = np.linalg.solve(D, R)
    dynamic_moments[code] = (D, R)
    dynamic_ir[code] = ir
    CY, CA = np.array(cluster_y), np.array(cluster_a)
    cluster_dynamic[code] = (CY, CA)
    rng = np.random.default_rng(2026)
    boot = []
    for b in range(240):
        weights = rng.poisson(1., len(ids))
        db = np.einsum("s,hsl->hl", weights, CA)
        rb = CY @ weights
        boot.append(np.linalg.solve(db, rb))
    boot = np.array(boot)
    print("\n", code, "estimated response by hours since a one-hour dose increment:")
    print("point:", np.round(ir, 4).tolist())
    print("cluster-bootstrap SD:", np.round(boot.std(axis=0), 4).tolist())
    print("ratios to first hour:", np.round(ir / ir[0], 4).tolist())

print("\n=== RATE-REGIME CHECKS, ONE-HOUR DML ===")
for code, j, cutpoints in [
    ("glucose_lab", 0, [0, 1.25, 2.25, np.inf]),
    ("map_arterial", 1, [0, .25, .75, np.inf])
]:
    ry = outcome_residuals[code]
    for lo, hi in zip(cutpoints[:-1], cutpoints[1:]):
        mm = np.isfinite(ry) & (Tc[:, j] > lo) & (Tc[:, j] <= hi)
        b, se = residual_cluster_fit(dose_resid[mm], ry[mm], gc[mm])
        print(code, (lo, hi), "n", mm.sum(),
              "slope", round(b[j], 4), "SE", round(se[j], 4))
print("elapsed seconds", round(time.time()-start, 1))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 6: dynamic identification\n"
        "- Cross-fitted immediate effects: glucose finger -1.9347 (SE .0445), lab -2.0148 (.0747); arterial MAP +3.0526 (.0887), cuff +2.5282 (.1734).\n"
        "- Estimated treatment impulse responses using dose innovations and removed their induced future-dose changes by moment deconvolution.\n"
        "- Used independent/randomly sampled glucose_lab for glucose dynamics and arterial MAP for pressure dynamics, with stay-cluster bootstrap.\n"
        f"- Impulse estimates: {dynamic_ir!r}\n"
        "- Interpretation depends on residual treatment variation being conditionally exogenous; dose-regime checks assess linearity within observed support.\n"
        f"- Python runs: {run_count}; forward passes: {forward_count}.\n"
    )
```
