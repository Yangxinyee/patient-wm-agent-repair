Infusions start five minutes after hourly readings, and doses are discretized. That gives a clear treatment–outcome ordering, but confounding remains possible. I’ll compare raw associations with estimates adjusted for preceding readings and doses. A large change after adjustment would show why the raw association is unreliable.

```python
PYTHON_RUNS += 1
from scipy import stats
from threadpoolctl import threadpool_limits
threadpool_limits(limits=8)

MEASUREMENTS = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
DRUGS = ["insulin", "norepinephrine"]

def make_panel(df):
    d = df.copy()
    d["t"] = np.floor(d.hour + 1e-5).astype(int)
    ids = np.sort(d.stay_id.unique())
    index = pd.MultiIndex.from_product([ids, np.arange(-1, 25)],
                                      names=["stay_id", "t"])
    m = d[d.event.isin(MEASUREMENTS)].pivot(
        index=["stay_id", "t"], columns="event", values="value")
    a = d[d.event.isin(DRUGS)].pivot(
        index=["stay_id", "t"], columns="event", values="rate")
    p = m.reindex(index).reindex(columns=MEASUREMENTS)
    p[DRUGS] = a.reindex(index).reindex(columns=DRUGS).fillna(0)
    return p

panel = make_panel(train)
panel_tuning = make_panel(tuning)
probe_ids = list(range(2020000000, 2020000100))
validation_ids = list(range(2020000200, 2020000300))

print("=== SIMULTANEOUS SENSOR DIFFERENCES ===")
for a, b in [("glucose_finger", "glucose_lab"), ("map_cuff", "map_arterial")]:
    delta = (panel[a] - panel[b]).dropna()
    print(a, "-", b, "n =", len(delta),
          "mean/std =", np.round([delta.mean(), delta.std()], 4),
          "quantiles =", delta.quantile([.05, .5, .95]).round(3).tolist())

print("\n=== DOSE DISTRIBUTIONS (including absent = zero) ===")
for drug in DRUGS:
    s = panel.loc[panel.index.get_level_values("t") >= 0, drug]
    print(drug, "quantiles:", s.quantile([0, .25, .5, .75, .9, .99, 1]).to_dict())
    print("Most frequent:", s.value_counts().head(12).to_dict())

# Features contain only information available before the hour's infusion.
g = panel.groupby(level="stay_id", sort=False)
last = g[MEASUREMENTS].ffill()
lag1 = last.groupby(level="stay_id", sort=False).shift(1)
lag2 = last.groupby(level="stay_id", sort=False).shift(2)
controls = pd.DataFrame(index=panel.index)
for c in MEASUREMENTS:
    controls[c] = last[c]
    controls[c + "_previous"] = lag1[c]
    controls[c + "_previous2"] = lag2[c]
    controls[c + "_observed"] = panel[c].notna().astype(float)
    controls[c + "_square"] = (last[c] / 100.) ** 2
for c in DRUGS:
    controls[c + "_previous"] = g[c].shift(1).fillna(0)
    controls[c + "_previous2"] = g[c].shift(2).fillna(0)
controls["hour"] = panel.index.get_level_values("t").astype(float)
controls["hour_square"] = (controls["hour"] / 24.) ** 2
controls = controls.fillna(controls.median())

def regression_dose_summary(target, drug):
    future = g[target].shift(-1)
    mask = (future.notna() &
            (controls.hour >= 0) & (controls.hour < 24))
    y = future[mask].to_numpy()
    dose = panel.loc[mask, drug].to_numpy()
    ids = panel.index.get_level_values("stay_id")[mask].to_numpy()
    X = controls.loc[mask].to_numpy(dtype=float)
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-8)
    X = np.column_stack([np.ones(len(y)), X, dose])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    inv = np.linalg.pinv(X.T @ X)
    influence = (X @ inv[:, -1]) * resid
    _, gi = np.unique(ids, return_inverse=True)
    cluster_score = np.bincount(gi, weights=influence)
    se = np.sqrt(np.sum(cluster_score ** 2))
    naive = np.linalg.lstsq(np.column_stack([np.ones(len(y)), dose]),
                           y, rcond=None)[0][-1]
    return dict(n=len(y), raw_slope=naive, adjusted_slope=coef[-1],
                cluster_se=se, residual_rmse=np.sqrt(np.mean(resid**2)))

print("\n=== INITIAL ASSOCIATIONS: NEXT READING PER UNIT RATE ===")
initial_associations = {}
for target, drug in [("glucose_finger", "insulin"),
                     ("glucose_lab", "insulin"),
                     ("map_arterial", "norepinephrine"),
                     ("map_cuff", "norepinephrine")]:
    result = regression_dose_summary(target, drug)
    initial_associations[(target, drug)] = result
    print(target, drug, {k: round(v, 5) for k, v in result.items()})

print("\n=== DOSE CONDITIONAL ON CURRENT PRIMARY READING ===")
for target, drug, width in [("glucose_finger", "insulin", 20),
                             ("map_arterial", "norepinephrine", 10)]:
    s = panel.loc[(panel.index.get_level_values("t") >= 0) &
                  panel[target].notna(), [target, drug]].copy()
    s["reading_band"] = np.floor(s[target] / width) * width
    print(target)
    print(s.groupby("reading_band")[drug].agg(["size", "mean", "std"])
          .round(4).to_string())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 3: hourly panels and preliminary regressions\n"
        "- Infusions begin at hour + 0.0833, after readings, and run one hour.\n"
        "- Built complete -1..24 panels; absent infusion means rate zero.\n"
        "- Preliminary adjusted slopes are associations, not yet established "
        "causal effects. Need nonlinear adjustment and independent checks.\n"
        f"- Sensor/dose summaries printed. Preliminary estimates: {initial_associations!r}\n"
        "- Probe stays: tuning IDs 2020000000..2020000099; independent validation "
        "IDs 2020000200..2020000299.\n"
        "- Python runs: 3; forward passes: 0; no edits.\n"
    )
```
