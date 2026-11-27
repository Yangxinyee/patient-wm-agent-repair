Simultaneous glucose readings closely agree, and simultaneous MAP readings share a strong signal with a roughly 3-unit offset. The different overall modality distributions therefore appear largely selection-related. I’ll estimate next-hour effects using dose variation unexplained by prior history. Stable estimates across held-out stays and exposure groups would support a causal interpretation; instability would argue for more controls.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)

def causal_features(p):
    g = p[["glucose_finger", "glucose_lab"]].mean(axis=1)
    m = pd.concat([p.map_arterial, p.map_cuff - 3.0], axis=1).mean(axis=1)
    x = pd.DataFrame(index=p.index)
    x["hour"] = p.index.get_level_values("t").to_numpy()
    for name in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        v = p[name]
        x[name] = v
        x[name + "_last"] = v.groupby(level=0).ffill()
        x[name + "_seen"] = v.notna().astype(float)
    for name, v in [("G", g), ("M", m)]:
        x[name] = v
        last = v.groupby(level=0).ffill()
        x[name+"_last"] = last
        count = v.notna().astype(float).groupby(level=0).cumsum()
        total = v.fillna(0).groupby(level=0).cumsum()
        x[name+"_mean_so_far"] = total / count.replace(0, np.nan)
        for lag in [1, 2, 3, 4]:
            x[name+"_lag"+str(lag)] = v.groupby(level=0).shift(lag)
            x[name+"_last_lag"+str(lag)] = last.groupby(level=0).shift(lag)
    for drug in ["insulin", "norepinephrine"]:
        x[drug+"_present"] = (p[drug] > 0).astype(float)
        for lag in [1, 2, 3, 4]:
            x[drug+"_lag"+str(lag)] = p[drug].groupby(level=0).shift(lag)
        x[drug+"_past_total"] = p[drug].groupby(level=0).cumsum()-p[drug]
    y = pd.DataFrame({
        "G_next": g.groupby(level=0).shift(-1),
        "M_next": m.groupby(level=0).shift(-1)
    })
    return x.astype(np.float32), y, g, m

X, Y, G, M = causal_features(panel)
rng = np.random.default_rng(3817)
all_train_ids = np.sort(train.stay_id.unique())
shuffled = rng.permutation(all_train_ids)
nuisance_ids = shuffled[:12000]
effect_test_ids = shuffled[12000:18000]
sid = panel.index.get_level_values("stay_id").to_numpy()
hours = panel.index.get_level_values("t").to_numpy()
fitmask = np.isin(sid, nuisance_ids) & (hours >= 1) & (hours <= 22)
testmask = np.isin(sid, effect_test_ids) & (hours >= 1) & (hours <= 22)

nuisance_models = {}
residuals = {}
for target, values in [
    ("insulin", panel.insulin), ("norepinephrine", panel.norepinephrine),
    ("G_next", Y.G_next), ("M_next", Y.M_next)
]:
    goodfit = fitmask & values.notna().to_numpy()
    model = HistGradientBoostingRegressor(
        max_iter=120, max_leaf_nodes=31, min_samples_leaf=100,
        l2_regularization=5.0, learning_rate=0.08,
        early_stopping=False, random_state=17
    )
    start = time.time()
    model.fit(X.loc[goodfit], values.loc[goodfit])
    pred = model.predict(X.loc[testmask])
    actual = values.loc[testmask].to_numpy()
    residuals[target] = actual-pred
    nuisance_models[target] = model
    finite = np.isfinite(actual)
    print(target, "fit_seconds", round(time.time()-start,1),
          "heldout_R2", round(r2_score(actual[finite], pred[finite]),4),
          "residual_SD", round(np.nanstd(actual-pred),4))

def clustered_ols(A, y, groups):
    good = np.isfinite(y) & np.isfinite(A).all(axis=1)
    a = np.column_stack([np.ones(good.sum()), A[good]])
    yy = y[good]
    beta = np.linalg.lstsq(a, yy, rcond=None)[0]
    err = yy-a@beta
    _, gi = np.unique(groups[good], return_inverse=True)
    sums = np.zeros((gi.max()+1, a.shape[1]))
    np.add.at(sums, gi, a*err[:,None])
    bread = np.linalg.pinv(a.T@a)
    cov = bread@(sums.T@sums)@bread
    se = np.sqrt(np.maximum(np.diag(cov),0))
    return beta, se, good.sum()

Ar = np.column_stack([residuals["insulin"], residuals["norepinephrine"]])
testsid = sid[testmask]
testpanel = panel.loc[testmask]
print("\n=== HELD-OUT RESIDUALIZED EFFECTS: NEXT HOUR PER UNIT OF RATE ===")
for outcome in ["G_next", "M_next"]:
    for label, mask in [
        ("all", np.ones(len(testpanel), dtype=bool)),
        ("insulin-positive", testpanel.insulin.to_numpy()>0),
        ("norepinephrine-positive", testpanel.norepinephrine.to_numpy()>0),
        ("hour<=12", hours[testmask]<=12),
        ("hour>12", hours[testmask]>12)
    ]:
        beta, se, n = clustered_ols(Ar[mask], residuals[outcome][mask], testsid[mask])
        print(outcome, label, "n", n,
              "insulin", round(beta[1],4), "+/-", round(1.96*se[1],4),
              "norepinephrine", round(beta[2],4), "+/-", round(1.96*se[2],4))

eval_ids_A = list(map(int, np.sort(tuning.stay_id.unique())[:100]))
eval_ids_B = list(map(int, np.sort(tuning.stay_id.unique())[100:200]))
print("\nEvaluation group A exposure fractions:",
      panel_tune.loc[(eval_ids_A, slice(0,24)), ["insulin","norepinephrine"]].gt(0).mean().to_dict())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Preliminary causal analysis (run 4)\n"
        "- Simultaneous glucose sensors correlate 0.974; median difference 0.\n"
        "- Simultaneous MAP sensors correlate 0.900; median arterial minus cuff is -3.\n"
        "- Built prior-history features and held-out residualized next-hour regressions.\n"
        "- Nuisances fitted on 12,000 stays; effects estimated on 6,000 different stays.\n"
        "- These are preliminary conditional estimates, not proof of identification.\n"
    )
    for outcome in ["G_next", "M_next"]:
        b,s,n = clustered_ols(Ar, residuals[outcome], testsid)
        f.write(f"- {outcome}: insulin {b[1]:.5f} (SE {s[1]:.5f}); "
                f"norepinephrine {b[2]:.5f} (SE {s[2]:.5f}); n={n}.\n")
```
