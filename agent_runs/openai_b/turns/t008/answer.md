The original factual NLL is 1.50467 on these stays. Before editing, I’ll use held-out, nonlinear history adjustment to revisit the treatment effects, especially norepinephrine. If its remaining association is treatment-selection bias, it should shrink when the dose and outcome models better account for past readings and titration.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)

hours = panel.index.get_level_values("hour").to_numpy()
stay_ids = panel.index.get_level_values("stay_id").to_numpy()
features = {"hour": hours.astype(float), "hour2": (hours/24.)**2}

for e in measurement_events:
    v = panel[e]
    present = v.notna()
    grp = v.groupby(level=0)
    features[e+"_now"] = v.to_numpy()
    for lag in [1,2,3]:
        features[e+"_lag"+str(lag)] = grp.shift(lag).to_numpy()
    filled = grp.ffill()
    features[e+"_last_before"] = filled.groupby(level=0).shift(1).to_numpy()
    observed_time = pd.Series(np.where(present, hours, np.nan), index=panel.index)
    last_time = observed_time.groupby(level=0).ffill().groupby(level=0).shift(1)
    features[e+"_age"] = hours-last_time.to_numpy()
    csum = v.fillna(0).groupby(level=0).cumsum()
    count = present.astype(float).groupby(level=0).cumsum()
    features[e+"_mean_before"] = (
        (csum-v.fillna(0))/(count-present.astype(float)).replace(0,np.nan)
    ).to_numpy()
    features[e+"_past_count"] = (count-present.astype(float)).to_numpy()
    initial = panel.xs(-1,level="hour")[e]
    features[e+"_baseline"] = initial.reindex(stay_ids).to_numpy()

for drug in drug_events:
    v = panel[drug]
    for lag in [1,2,3]:
        features[drug+"_lag"+str(lag)] = v.groupby(level=0).shift(lag).to_numpy()
    csum = v.groupby(level=0).cumsum()
    features[drug+"_mean_before"] = (
        (csum-v)/np.maximum(hours+1,1)
    ).to_numpy()

Xhistory = pd.DataFrame(features, index=panel.index).astype(np.float32)
causal_results = {}

for drug, outcome in [("insulin","glucose_finger"),
                       ("norepinephrine","map_arterial")]:
    mask = ((hours >= 0) & (hours <= 23) &
            panel[outcome].notna().to_numpy() &
            panel[outcome+"_next"].notna().to_numpy())
    X = Xhistory.loc[mask].copy()
    other = "norepinephrine" if drug == "insulin" else "insulin"
    X["other_current_dose"] = panel.loc[mask,other].to_numpy()
    r = panel.loc[mask,drug].to_numpy()
    dy = (panel.loc[mask,outcome+"_next"] -
          panel.loc[mask,outcome]).to_numpy()
    ids = stay_ids[mask]
    folds = ids % 2
    rhat = np.empty(len(r))
    yhat = np.empty(len(r))
    started = time.time()
    for fold in [0,1]:
        tr, te = folds != fold, folds == fold
        params = dict(
            max_iter=180, learning_rate=.07, max_leaf_nodes=31,
            min_samples_leaf=120, l2_regularization=10.,
            early_stopping=False, random_state=410+fold
        )
        rm = HistGradientBoostingRegressor(**params).fit(X.iloc[tr],r[tr])
        ym = HistGradientBoostingRegressor(**params).fit(X.iloc[tr],dy[tr])
        rhat[te] = rm.predict(X.iloc[te])
        yhat[te] = ym.predict(X.iloc[te])
        print(drug, "fold", fold,
              "dose RMSE", round(np.sqrt(np.mean((r[te]-rhat[te])**2)),4),
              "change RMSE", round(np.sqrt(np.mean((dy[te]-yhat[te])**2)),4))
    rr, yr = r-rhat, dy-yhat
    beta = rr@yr/(rr@rr)
    scores = pd.Series(rr*(yr-beta*rr)).groupby(ids).sum().to_numpy()
    se = np.sqrt(scores@scores)/(rr@rr)
    causal_results[drug] = {
        "beta": float(beta), "cluster_se": float(se),
        "n": len(r), "residual_rate_sd": float(rr.std())
    }
    print("CROSSFIT RESULT:", drug, causal_results[drug],
          "seconds", round(time.time()-started,1))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 4 and baseline forwards\n"
        "Insulin exact-current FE slope 0.034 (stay-cluster SE 0.0426); "
        "rich history-adjusted slope 0.0105, lag -0.0113. "
        "Norepinephrine exact-current slope -2.5509; rich history slope "
        "-0.3866, lag +0.117. Norepinephrine titration is persistent and "
        "some contemporaneous sensors are missing, so remaining confounding "
        "is plausible; no final conclusion yet.\n"
        "Forward 1 (files fwd_2; one rejected duplicate-ID request before it): "
        "original tuning first100 rollouts insulin x2 +0.209, half -0.072; "
        "NE x2 -0.007, half +0.032. Forward 2 (fwd_3): original factual "
        "NLL 1.50467 on 4,822 readings.\n"
        "Run 5 cross-fitted nonlinear pre-treatment-history estimates: "
        +json.dumps(causal_results)+"\n"
    )
```