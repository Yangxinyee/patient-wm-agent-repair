State extraction failed for the same missing-file reason, so no forward mode has worked. I’ll test delayed effects and sensitivity to richer history models using the records. Near-zero responses across horizons would support removing dose sensitivity; a reproducible delayed response would require preserving it.

```python
t0 = time.time()
effect_reports = []
long_crossfit = {}

with threadpool_limits(limits=8):
    for drug, outcome, baseline, fused in [
        ("insulin", "glucose_finger", Glast, Gsame),
        ("norepinephrine", "map_arterial", Mlast, Msame)
    ]:
        other = "norepinephrine" if drug == "insulin" else "insulin"
        X = np.column_stack([Xhistory, arr[other][:, ci].reshape(-1)]).astype(np.float32)
        d = arr[drug][:, ci].reshape(-1).astype(float)

        # Reuse held-out dose models, predicting all hours rather than only
        # hours with the original one-hour outcome.
        pd_all = np.full(len(d), np.nan)
        for fold in [0, 1]:
            use = fold_rows == fold
            pd_all[use] = crossfit[drug]["fitted"][fold][0].predict(X[use])
        dr = d - pd_all

        for horizon in [1, 2, 3, 4, 6, 8]:
            yn = np.full((nstay, len(ci)), np.nan)
            good_col = ci + horizon < 26
            yn[:, good_col] = (
                arr[outcome][:, ci[good_col]+horizon] - baseline[:, ci[good_col]]
            )
            y = yn.reshape(-1)
            valid = np.isfinite(y)
            py = np.full(len(y), np.nan)
            for fold in [0, 1]:
                tr = valid & (fold_rows != fold)
                te = valid & (fold_rows == fold)
                my = HistGradientBoostingRegressor(
                    max_iter=180, max_leaf_nodes=23, min_samples_leaf=150,
                    learning_rate=.07, l2_regularization=20,
                    early_stopping=False, random_state=200+fold)
                my.fit(X[tr], y[tr])
                py[te] = my.predict(X[te])
            yr = y-py
            long_crossfit[(drug, horizon)] = dict(dr=dr, yr=yr, valid=valid)
            for label, sub in [
                ("all", valid),
                ("h12+", valid & (hour_index >= 12))
            ]:
                b, se = cluster_slope(dr[sub], yr[sub], sid_index[sub])
                effect_reports.append([drug, "primary", horizon, label, b, se, sub.sum()])

        # Independent sensor / combined-sensor sensitivity at one hour.
        y = (fused[:, ci+1] - baseline[:, ci]).reshape(-1)
        valid = np.isfinite(y)
        py = np.full(len(y), np.nan)
        for fold in [0, 1]:
            tr, te = valid & (fold_rows != fold), valid & (fold_rows == fold)
            my = HistGradientBoostingRegressor(
                max_iter=180, max_leaf_nodes=23, min_samples_leaf=150,
                learning_rate=.07, l2_regularization=20,
                early_stopping=False, random_state=300+fold)
            my.fit(X[tr], y[tr])
            py[te] = my.predict(X[te])
        b, se = cluster_slope(dr[valid], (y-py)[valid], sid_index[valid])
        effect_reports.append([drug, "combined sensors", 1, "all", b, se, valid.sum()])

        # Refit both nuisances with more capacity.
        y = crossfit[drug]["y"]
        valid = crossfit[drug]["valid"]
        pd_rich, py_rich = np.full(len(d), np.nan), np.full(len(d), np.nan)
        for fold in [0, 1]:
            tr, te = valid & (fold_rows != fold), valid & (fold_rows == fold)
            kw = dict(max_iter=400, max_leaf_nodes=31, min_samples_leaf=100,
                      learning_rate=.05, l2_regularization=20,
                      early_stopping=False, random_state=400+fold)
            md, my = HistGradientBoostingRegressor(**kw), HistGradientBoostingRegressor(**kw)
            md.fit(X[tr], d[tr])
            my.fit(X[tr], y[tr])
            pd_rich[te], py_rich[te] = md.predict(X[te]), my.predict(X[te])
        for label, sub in [("all", valid), ("h12+", valid & (hour_index >= 12))]:
            b, se = cluster_slope((d-pd_rich)[sub], (y-py_rich)[sub], sid_index[sub])
            effect_reports.append([drug, "richer nuisances", 1, label, b, se, sub.sum()])
        print("Completed", drug, "elapsed_s", round(time.time()-t0, 1))

effect_table = pd.DataFrame(effect_reports,
    columns=["drug", "analysis", "horizon", "hours", "coef", "cluster_se", "N"])
print(effect_table.round(5).to_string(index=False))

print("\n=== NUMERIC VOCAB ===")
print(repr(VOCAB["numeric"])[:6500])
print("\n=== ALL NON-EMBEDDING TENSOR SHAPES ===")
for n in NAMES:
    if not n.startswith("embed."):
        print(n, SHAPES[n])

with open("/work/notes.md", "a") as f:
    f.write("\n## Horizon and nuisance checks\nState forward also failed on missing cuts_tuning.parquet. One-hour cross-fit estimates: insulin 0.00139 (SE 0.03899), norepinephrine 0.19499 (SE 0.08746) per rate unit. Run 10 tests horizons 1-8, combined sensors, and richer nuisance fits; full results retained in effect_table. No model edits. Python runs: 10; successful forward passes: 0.\n")
    f.write(effect_table.round(5).to_string(index=False) + "\n")
```