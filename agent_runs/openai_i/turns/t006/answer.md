The original rollouts show small insulin responses and an asymmetric norepinephrine response. That does not resolve the disagreement in the record-based estimates. I’ll now use stay-separated, nonlinear models to isolate dose variation unexplained by recorded history. Stable residual-dose effects across time and sensors would strengthen the causal case; disappearance would suggest the linear estimates were misleading.

```python
RUN_COUNTS["python"] += 1
RUN_COUNTS["forward"] = 2

import time
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)

factual_base = pd.read_csv("/work/fwd_1/factual_original.csv")
dose_base = pd.read_csv("/work/fwd_2/dose_edit_original.csv")
print("Factual file:", list(factual_base), factual_base.head(2).to_dict("records"))
print("Dose file:", list(dose_base), dose_base.head(2).to_dict("records"))
print("Note: baseline factual used 100 stays; baseline dose request used 99.")

HC = H.copy()
for sensor in MEAS:
    for lag in (3, 4):
        HC[sensor + f"_lag{lag}"] = (
            last[sensor].groupby(level="stay_id").shift(lag)
            .fillna(medians[sensor])
        )
    obs = panel[sensor].notna().astype(float)
    prior_count = obs.groupby(level="stay_id").cumsum() - obs
    prior_sum = (
        panel[sensor].fillna(0).groupby(level="stay_id").cumsum()
        - panel[sensor].fillna(0)
    )
    HC[sensor + "_past_mean"] = (
        prior_sum / prior_count.replace(0, np.nan)
    ).fillna(medians[sensor])
for drug in DRUGS:
    for lag in (3, 4):
        HC[drug + f"_lag{lag}"] = (
            panel[drug].groupby(level="stay_id").shift(lag).fillna(0)
        )
    HC[drug + "_past_mean"] = (
        panel[drug].groupby(level="stay_id").cumsum() - panel[drug]
    ) / np.maximum(tt, 1)

train_ids = np.sort(train.stay_id.unique())
fold_rng = np.random.default_rng(1729)
fold1_ids = fold_rng.choice(train_ids, len(train_ids) // 2, replace=False)
fold = np.isin(stay_array, fold1_ids).astype(np.int8)
valid_time = (tt >= 0) & (tt <= 23)

def new_nuisance(seed):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06, max_iter=140,
        max_leaf_nodes=23, min_samples_leaf=80, l2_regularization=10,
        early_stopping=False, random_state=seed
    )

def innovation_slope(ra, ry, groups):
    denom = float(ra @ ra)
    beta = float(ra @ ry) / denom
    _, gi = np.unique(groups, return_inverse=True)
    scores = np.bincount(gi, weights=ra * (ry - beta * ra))
    se = float(np.sqrt(np.sum(scores**2)) / denom)
    return beta, se

dose_models, outcome_models = {}, {}
dose_residuals, outcome_residuals = {}, {}
DML = []
start = time.time()

for di, drug in enumerate(DRUGS):
    other = DRUGS[1 - di]
    x = np.column_stack([
        HC.to_numpy(dtype=np.float32),
        panel[other].to_numpy(dtype=np.float32)
    ])
    a = panel[drug].to_numpy(dtype=np.float64)
    active = valid_time & (a > 0)
    ahat = np.full(len(panel), np.nan)
    dose_models[drug] = {}
    for k in (0, 1):
        tr = active & (fold != k)
        te = active & (fold == k)
        model = new_nuisance(100 + 10 * di + k)
        model.fit(x[tr], a[tr])
        ahat[te] = model.predict(x[te])
        dose_models[drug][k] = model
    ra = a - ahat
    dose_residuals[drug] = ra
    r2 = 1 - np.mean(ra[active]**2) / np.var(a[active])
    print("\nPolicy", drug, "out-of-fold R2=", round(float(r2), 4),
          "dose SD=", round(float(a[active].std()), 4),
          "innovation SD=", round(float(ra[active].std()), 4),
          "elapsed seconds=", round(time.time() - start, 1))

    sensors = MEAS[:2] if drug == "insulin" else MEAS[2:]
    paired_current = panel[sensors].notna().all(axis=1).to_numpy()
    for si, sensor in enumerate(sensors):
        y = panel[sensor].groupby(level="stay_id").shift(-1).to_numpy()
        eligible = active & np.isfinite(y)
        yhat = np.full(len(panel), np.nan)
        outcome_models[sensor] = {}
        for k in (0, 1):
            tr = eligible & (fold != k)
            te = eligible & (fold == k)
            model = new_nuisance(300 + 10 * di + 2 * si + k)
            model.fit(x[tr], y[tr])
            yhat[te] = model.predict(x[te])
            outcome_models[sensor][k] = model
        ry = y - yhat
        outcome_residuals[sensor] = ry
        subsets = {
            "all active": eligible,
            "hour 0": eligible & (tt == 0),
            "hours 1-5": eligible & (tt >= 1) & (tt <= 5),
            "hours 6-11": eligible & (tt >= 6) & (tt <= 11),
            "hours 12-23": eligible & (tt >= 12),
            "both sensors now": eligible & paired_current
        }
        for label, mask in subsets.items():
            if mask.sum() >= 300:
                beta, se = innovation_slope(
                    ra[mask], ry[mask], stay_array[mask]
                )
                DML.append({
                    "sensor": sensor, "subset": label,
                    "n": int(mask.sum()), "beta": beta, "cluster_se": se
                })

dml_table = pd.DataFrame(DML)
print("\n=== CROSSFITTED DOSE-INNOVATION ESTIMATES ===")
print(dml_table.round(4).to_string(index=False))
print("Elapsed seconds:", round(time.time() - start, 1))

print("\n=== HOUR-0 DOSE POLICY, COARSE BINS ===")
p0 = panel.xs(0, level="t")
for drug, sensor, width in [
    ("insulin", "glucose_finger", 10),
    ("norepinephrine", "map_cuff", 5)
]:
    z = p0.loc[p0[sensor].notna(), [sensor, drug]].copy()
    z["reading_bin"] = np.floor(z[sensor] / width) * width
    tab = z.groupby("reading_bin").agg(
        n=(drug, "size"), mean_rate=(drug, "mean"),
        active_fraction=(drug, lambda v: (v > 0).mean())
    )
    print(drug)
    print(tab.loc[tab.n >= 100].round(3).to_string())

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Nonlinear identification — Python 4; forward 2\n"
        "- Original development factual NLL 1.66206; limit 1.66706.\n"
        "- Original rollouts (99 stays): insulin x2 -1.864, half +0.975; "
        "NE x2 +0.342, half -0.032.\n"
        "- Crossfit dose and outcome nuisance models by stay, using only "
        "pre-dose history. Active-dose estimates still require exchangeability "
        "and attention to observation selection and nuisance-model error.\n"
        + dml_table.to_string(index=False) + "\n"
    )
print("\nCounts:", RUN_COUNTS, "Current edits:", edited())
```
