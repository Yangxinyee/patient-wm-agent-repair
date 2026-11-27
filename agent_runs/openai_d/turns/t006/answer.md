The original rollout predicts insulin doubling lowers glucose by 1.847, while both norepinephrine doubling and halving lower MAP. Those responses alone do not tell us what is correct. I’ll fit nonlinear, stay-held-out models of dose choice and next readings, focusing on active infusions. Stable residual-dose estimates would strengthen the evidence; further shifts would expose remaining confounding.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import time

threadpool_limits(8)
t0_run = time.time()
tt_all = wide.index.get_level_values("t").to_numpy(float)
Xhist = pd.DataFrame(index=wide.index)
Xhist["hour"] = tt_all

for c in measurements:
    Xhist[c+"_observed"] = wide[c].notna().astype(float)
    Xhist[c+"_current"] = wide[c]
    Xhist[c+"_last"] = wide[c+"_last"]
    obs_time = pd.Series(tt_all, index=wide.index).where(wide[c].notna())
    last_time = obs_time.groupby(level=0).ffill()
    Xhist[c+"_age"] = tt_all - last_time.to_numpy()
    for lag in (1, 2, 3):
        Xhist[c+f"_lag{lag}"] = wide[c+f"_lag{lag}"]
    Xhist[c+"_change1"] = wide[c+"_last"] - wide[c+"_lag1"]
    Xhist[c+"_change2"] = wide[c+"_lag1"] - wide[c+"_lag2"]
    # Online smoothing: no future readings enter these features.
    arr = wide[c].to_numpy().reshape(len(train_ids), 26)
    sm = np.full_like(arr, np.nan)
    state = np.full(len(train_ids), np.nan)
    for j in range(26):
        observed = np.isfinite(arr[:, j])
        initialise = observed & ~np.isfinite(state)
        update = observed & np.isfinite(state)
        state[initialise] = arr[initialise, j]
        state[update] = .4 * arr[update, j] + .6 * state[update]
        sm[:, j] = state
    Xhist[c+"_smooth"] = sm.ravel()
    initial = np.where(np.isfinite(arr[:, 0]), arr[:, 0], arr[:, 1])
    Xhist[c+"_baseline"] = np.repeat(initial, 26)

for c in drugs:
    Xhist[c+"_active"] = (wide[c] > 0).astype(float)
    for lag in (1, 2, 3):
        Xhist[c+f"_lag{lag}"] = wide[c+f"_lag{lag}"]
    Xhist[c+"_past_change"] = wide[c+"_lag1"] - wide[c+"_lag2"]
Xhist = Xhist.astype(np.float32)

def residual_effect(ry, rd, groups):
    den = rd @ rd
    b = (rd @ ry) / den
    err = ry - b * rd
    lab = pd.factorize(groups)[0]
    score = np.bincount(lab, weights=rd * err)
    se = np.sqrt(score @ score) / den
    return round(float(b), 5), round(float(se), 5), round(float(np.std(rd)), 5), len(ry)

dml_results = {}
dml_cache = {}
for outcome, drug, other in [
    ("glucose_finger", "insulin", "norepinephrine"),
    ("map_arterial", "norepinephrine", "insulin"),
]:
    valid = (
        (tt_all >= 1) & (tt_all <= 23)
        & wide[outcome+"_next"].notna().to_numpy()
        & wide[outcome+"_last"].notna().to_numpy()
        & (wide[drug].to_numpy() > 0)
    )
    inds = np.flatnonzero(valid)
    X = Xhist.iloc[inds].copy()
    X[other+"_current_rate"] = wide[other].iloc[inds].to_numpy()
    X = X.to_numpy()
    y = (wide[outcome+"_next"] - wide[outcome+"_last"]).iloc[inds].to_numpy()
    d = wide[drug].iloc[inds].to_numpy()
    groups = wide.index.get_level_values("stay_id").to_numpy()[inds]
    hours = tt_all[inds]
    folds = ((groups - train_ids[0]) % 2).astype(int)
    pred_y = np.zeros_like(y)
    pred_d = np.zeros_like(d)
    models = []
    for fold in (0, 1):
        tr, va = folds != fold, folds == fold
        pair = []
        for target, destination, label in [
            (y, pred_y, "outcome"), (d, pred_d, "dose")
        ]:
            model = HistGradientBoostingRegressor(
                max_iter=160, max_leaf_nodes=31, learning_rate=.07,
                min_samples_leaf=100, l2_regularization=10.0,
                early_stopping=False, random_state=123
            )
            model.fit(X[tr], target[tr])
            destination[va] = model.predict(X[va])
            pair.append(model)
            print(drug, "fold", fold, label, "fit done; elapsed",
                  round(time.time() - t0_run, 1), "s", flush=True)
        models.append(pair)
    ry, rd = y - pred_y, d - pred_d
    print("\nDML", drug, "effect, clustered SE, residual dose SD, n:")
    result = {}
    subsets = {
        "all active": np.ones(len(y), dtype=bool),
        "hours >=12": hours >= 12,
        "lower half of doses": d <= np.median(d),
        "upper half of doses": d > np.median(d),
        "away from maximum": d < (14.5 if drug == "insulin" else 1.0),
        "current primary observed": wide[outcome].iloc[inds].notna().to_numpy()
    }
    for label, subset in subsets.items():
        if subset.sum() > 100:
            result[label] = residual_effect(ry[subset], rd[subset], groups[subset])
            print(label, result[label])
    print("Held-out dose R2:", round(1 - np.var(rd)/np.var(d), 5),
          "outcome innovation RMSE:", round(np.sqrt(np.mean(ry**2)), 4))
    dml_results[drug] = result
    dml_cache[drug] = dict(
        inds=inds, X=X, y=y, d=d, groups=groups, hours=hours,
        pred_y=pred_y, pred_d=pred_d, ry=ry, rd=rd, models=models
    )

print("\nTotal elapsed seconds:", round(time.time() - t0_run, 1))
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward requests 1 and 2\n"
        "Original factual NLL (tuning IDs 2020000000..99): 1.63461; "
        "provisional maximum 1.63961.\n"
        "Original rollout: insulin x2 -1.847, half +0.777; "
        "NE x2 -0.498, half -0.600.\n"
        "\n## Run 4\n"
        "Cross-fitted nonlinear nuisance regressions, split by stay, active "
        "infusions only. Features use current/past readings, observation ages, "
        "online smoothing, pre-ICU baseline, prior doses, time and other drug. "
        "Estimated residual-dose slopes and sensitivity across dose/hour subsets.\n"
        + json.dumps(dml_results, indent=2) + "\n"
    )
```