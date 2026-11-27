Doses are recorded five minutes after the hourly readings, so we can compare subsequent changes while controlling for information recorded before treatment. Raw dose–outcome correlations would still be confounded. I’ll build hourly histories and check whether estimated effects remain stable as I add prior readings and doses; large changes would argue against trusting a simple regression.

```python
from scipy.linalg import lstsq

print("Training hour range:", train.hour.min(), train.hour.max())
print("Tuning IDs for baseline requests:", tuning_ids[:100].tolist())

measurements = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
drugs = ["insulin", "norepinephrine"]
idx = pd.MultiIndex.from_product(
    [train_ids, np.arange(-1, 25)], names=["stay_id", "t"]
)
wide = train.loc[train.event.isin(measurements)].pivot_table(
    index=["stay_id", "hour"], columns="event", values="value", aggfunc="first"
)
wide.index = wide.index.set_names(["stay_id", "t"])
wide = wide.reindex(idx)
actions = train.loc[train.event.isin(drugs), ["stay_id", "hour", "event", "rate"]].copy()
actions["t"] = np.floor(actions.hour).astype(int)
actions = actions.pivot_table(
    index=["stay_id", "t"], columns="event", values="rate", aggfunc="sum"
)
wide = wide.join(actions).sort_index()
wide[drugs] = wide[drugs].fillna(0.0)

for c in measurements:
    wide[c + "_last"] = wide[c].groupby(level=0).ffill()
    for lag in (1, 2, 3):
        wide[c + f"_lag{lag}"] = wide[c + "_last"].groupby(level=0).shift(lag)
    wide[c + "_next"] = wide[c].groupby(level=0).shift(-1)
for c in drugs:
    for lag in (1, 2, 3):
        wide[c + f"_lag{lag}"] = wide[c].groupby(level=0).shift(lag).fillna(0.0)

print("\nPaired assay differences:")
for p, s in [("glucose_finger", "glucose_lab"), ("map_arterial", "map_cuff")]:
    diff = wide[p] - wide[s]
    print(p, "-", s, diff.describe(percentiles=[.1, .5, .9]).round(3).to_dict())

def partial_linear(y, d, controls, groups):
    X = np.column_stack([np.ones(len(y)), controls])
    scales = np.sqrt(np.mean(X * X, axis=0))
    scales[scales < 1e-10] = 1.0
    X = X / scales
    sol = lstsq(X, np.column_stack([y, d]), lapack_driver="gelsy")[0]
    ry, rd = (np.column_stack([y, d]) - X @ sol).T
    denom = rd @ rd
    beta = (rd @ ry) / denom
    resid = ry - beta * rd
    labels = pd.factorize(groups)[0]
    scores = np.bincount(labels, weights=rd * resid)
    se = np.sqrt(scores @ scores) / denom
    return {
        "effect_per_rate_unit": round(float(beta), 5),
        "cluster_SE": round(float(se), 5),
        "residual_dose_SD": round(float(rd.std()), 5),
        "dose_R2": round(float(1 - np.var(rd) / np.var(d)), 4),
        "outcome_residual_SD": round(float(resid.std()), 3),
        "n": len(y)
    }

initial_effects = {}
for p, s, drug, other, cross in [
    ("glucose_finger", "glucose_lab", "insulin", "norepinephrine", "map_arterial"),
    ("map_arterial", "map_cuff", "norepinephrine", "insulin", "glucose_finger"),
]:
    cols = [p, p+"_next", p+"_lag1", p+"_lag2"]
    ok = wide[cols].notna().all(axis=1)
    tt = wide.index.get_level_values("t").to_numpy()
    ok &= (tt >= 1) & (tt <= 23)
    w = wide.loc[ok].copy()
    t = w.index.get_level_values("t").to_numpy(float)
    groups = w.index.get_level_values("stay_id").to_numpy()
    y = w[p+"_next"].to_numpy()
    d = w[drug].to_numpy()
    current = w[p].to_numpy()
    lag1 = w[p+"_lag1"].to_numpy()
    lag2 = w[p+"_lag2"].to_numpy()
    secondary = w[s+"_last"].fillna(w[p]).to_numpy()
    secondary_age_missing = w[s].isna().to_numpy(float)
    cross_current = w[cross+"_last"].fillna(w[cross+"_last"].median()).to_numpy()
    levels = [
        ("unadjusted", np.empty((len(w), 0))),
        ("current + time", np.column_stack([current, current**2, t, t**2])),
        ("+ reading history", np.column_stack([
            current, current**2, lag1, lag2, t, t**2
        ])),
        ("+ assays, other drug, past doses", np.column_stack([
            current, current**2, lag1, lag2, secondary, secondary_age_missing,
            cross_current, t, t**2, w[other],
            w[drug+"_lag1"], w[drug+"_lag2"],
            w[other+"_lag1"], w[other+"_lag2"]
        ]))
    ]
    print("\nOutcome:", p, "treatment:", drug)
    initial_effects[drug] = {}
    for label, X in levels:
        result = partial_linear(y, d, X, groups)
        initial_effects[drug][label] = result
        print(label, result)
    X = levels[-1][1]
    for label, subset in [
        ("active infusions only", d > 0),
        ("hours 12-23", t >= 12),
    ]:
        print(label, partial_linear(y[subset], d[subset], X[subset], groups[subset]))

print("\nDose summaries:")
print(wide.loc[wide.index.get_level_values("t") >= 0, drugs].describe().round(4).to_string())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 3\n"
        "Built complete hourly panel (-1..24); an absent infusion is zero rate. "
        "Readings are at integer hours, infusions at hour + 0.0833. "
        "Compared next-primary-reading regressions with increasingly rich "
        "pre-treatment controls; standard errors clustered by stay.\n"
        + json.dumps(initial_effects, indent=2) + "\n"
        "These preliminary regressions are diagnostic, not yet accepted as "
        "causal. Next: baseline model forward results and nonlinear/history "
        "robustness checks.\n"
    )
```