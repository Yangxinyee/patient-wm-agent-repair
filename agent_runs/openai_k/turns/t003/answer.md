The records show doses chosen shortly after hourly readings, with substantial variation in both drugs. That makes timing-based analysis possible, but a raw dose–outcome association would still be confounded. I’ll build hourly histories and compare unadjusted associations with history-adjusted estimates across both measurement types. Agreement across sensors and adjustments would strengthen the evidence; disagreement would flag confounding or measurement selection.

```python
run_count += 1
from sklearn.preprocessing import SplineTransformer

print("=== NUMERIC VOCAB ===")
for item in VOCAB["numeric"]:
    print({k: ({"n": len(v), "first": v[:3], "last": v[-3:]}
               if isinstance(v, list) and len(v) > 6 else v)
           for k, v in item.items()})
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])
print("tuning IDs, first 100:", tune.stay_id.unique()[:100].tolist())

measure_codes = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
drug_codes = ["insulin", "norepinephrine"]
ids = np.sort(df.stay_id.unique())
grid = pd.MultiIndex.from_product([ids, np.arange(-1, 25)],
                                  names=["stay_id", "hour"])
m = df.loc[df.event.isin(measure_codes), ["stay_id", "hour", "event", "value"]]
print("duplicate measurement cells:",
      m.duplicated(["stay_id", "hour", "event"]).sum())
P = m.pivot(index=["stay_id", "hour"], columns="event",
            values="value").reindex(grid)[measure_codes]
a = df.loc[df.event.isin(drug_codes),
           ["stay_id", "hour", "event", "rate"]].copy()
a["hour"] = np.floor(a.hour).astype(int)
rates = a.pivot(index=["stay_id", "hour"], columns="event",
                values="rate").reindex(grid)[drug_codes].fillna(0)
P = P.join(rates)
last = P[measure_codes].groupby(level=0, sort=False).ffill()
past = last.groupby(level=0, sort=False).shift(1)
previous_rate = rates.groupby(level=0, sort=False).shift(1).fillna(0)
next_values = P[measure_codes].groupby(level=0, sort=False).shift(-1)
hours = P.index.get_level_values("hour").to_numpy()
sid = P.index.get_level_values("stay_id").to_numpy()

print("\n=== CO-OBSERVED SENSOR DIFFERENCES ===")
for x, y in [("glucose_finger", "glucose_lab"), ("map_cuff", "map_arterial")]:
    d = (P[x] - P[y]).dropna()
    print(x, "-", y, "n", len(d), "mean", round(d.mean(), 4),
          "sd", round(d.std(), 4),
          "quantiles", d.quantile([.01, .5, .99]).round(2).to_dict())

base0 = P[measure_codes].xs(0, level="hour").combine_first(
    P[measure_codes].xs(-1, level="hour"))
base_values = base0.reindex(sid).to_numpy()
last_values = last.to_numpy()
past_values = past.to_numpy()
med = np.nanmedian(last_values, axis=0)
scale_m = np.array([40., 40., 15., 15.])
lv = (np.where(np.isfinite(last_values), last_values, med) - med) / scale_m
pv = (np.where(np.isfinite(past_values), past_values, med) - med) / scale_m
bv = (np.where(np.isfinite(base_values), base_values, med) - med) / scale_m
Xbasic = np.column_stack([
    np.ones(len(P)), hours / 24, (hours / 24)**2,
    lv, lv**2, pv, bv,
    np.isfinite(last_values).astype(float),
    previous_rate.to_numpy() / np.array([2., .5])
])
T_all = rates.to_numpy()
valid_time = (hours >= 0) & (hours <= 23)

def fwl_fit(X, T, y, groups):
    reg = 1e-7 * np.eye(X.shape[1])
    reg[0, 0] = 0
    xtx = X.T @ X + reg
    rr = T - X @ np.linalg.solve(xtx, X.T @ T)
    ry = y - X @ np.linalg.solve(xtx, X.T @ y)
    gram = rr.T @ rr
    beta = np.linalg.solve(gram, rr.T @ ry)
    err = ry - rr @ beta
    _, gi = np.unique(groups, return_inverse=True)
    scores = np.zeros((gi.max()+1, T.shape[1]))
    np.add.at(scores, gi, rr * err[:, None])
    bread = np.linalg.inv(gram)
    se = np.sqrt(np.diag(bread @ (scores.T @ scores) @ bread))
    return beta, se, np.sqrt(np.mean(rr**2, axis=0))

print("\n=== ONE-HOUR ASSOCIATIONS, NOT YET CAUSAL CONCLUSIONS ===")
preliminary = {}
for code in measure_codes:
    mask = valid_time & P[code].notna().to_numpy() & next_values[code].notna().to_numpy()
    y = next_values.loc[mask, code].to_numpy()
    cur = P.loc[mask, code].to_numpy()
    t = T_all[mask]
    raw_beta, raw_se, _ = fwl_fit(
        np.ones((len(y), 1)), t, y-cur, sid[mask])
    spline = SplineTransformer(n_knots=12, degree=3, include_bias=False)
    sx = spline.fit_transform(cur[:, None])
    X = np.column_stack([Xbasic[mask], sx])
    beta, se, residual_sd = fwl_fit(X, t, y, sid[mask])
    preliminary[code] = dict(n=len(y), beta=beta, se=se,
                             residual_dose_sd=residual_sd)
    print(code, "n", len(y),
          "raw delta slopes", np.round(raw_beta, 4).tolist(),
          "adjusted slopes", np.round(beta, 4).tolist(),
          "cluster SE", np.round(se, 4).tolist(),
          "residual dose SD", np.round(residual_sd, 4).tolist())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 3: hourly panel and preliminary associations\n"
        "- Doses occur at hour + 0.0833, after same-hour measurements; next-hour outcomes are post-treatment.\n"
        "- Constructed complete hourly histories, with zero rates for absent infusion records.\n"
        "- Computed raw and flexible history-adjusted one-hour slopes; these remain provisional because treatment and observation selection may confound them.\n"
        f"- Preliminary estimates: {preliminary!r}\n"
        f"- Python runs: {run_count}; forward passes: {forward_count}.\n"
    )
```
