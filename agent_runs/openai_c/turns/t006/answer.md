The factual baseline is 1.63461 nats per reading. Insulin dosing appears to jump at glucose thresholds, offering a stronger check than ordinary regression. I’ll test those jumps and use nonlinear, stay-separated adjustment for both drugs. Agreement across methods and sensors would support a causal estimate; disagreement would keep it uncertain.

```python
PYTHON_RUNS += 1
FORWARD_PASSES = 2
from sklearn.ensemble import HistGradientBoostingRegressor
import time

baseline_nll = 1.63461
baseline_rollouts = {
    "insulin_x2": -1.847, "insulin_half": 0.777,
    "ne_x2": -0.498, "ne_half": -0.600
}
print("=== FACTUAL FILE FORMAT ===")
baseline_factual = pd.read_csv("/work/fwd_2/factual_original.csv")
print(baseline_factual.head(3).to_string(index=False))

# Add strictly pre-infusion history features.
t = panel.index.get_level_values("t").to_numpy()
sid = panel.index.get_level_values("stay_id").to_numpy()
for c in MEASUREMENTS:
    observed_time = pd.Series(np.where(panel[c].notna(), t, np.nan),
                              index=panel.index)
    observed_time = observed_time.groupby(level="stay_id").ffill()
    controls[c + "_age"] = (t - observed_time).fillna(30)
    controls[c + "_change"] = (last[c] - lag1[c]).fillna(0)
    cs = panel[c].fillna(0).groupby(level="stay_id").cumsum()
    cn = panel[c].notna().astype(int).groupby(level="stay_id").cumsum()
    controls[c + "_past_mean"] = (cs / cn.replace(0, np.nan)).fillna(last[c])
    baseline = panel[c].where(t <= 0).groupby(level="stay_id").ffill()
    controls[c + "_baseline"] = baseline.fillna(panel[c].median())
for c in DRUGS:
    previous = g[c].shift(1).fillna(0)
    controls[c + "_past_mean"] = (
        previous.groupby(level="stay_id").cumsum() / np.maximum(t + 1, 1)
    )
controls = controls.fillna(controls.median()).fillna(0)

def clustered_ratio(z, d, y, groups):
    denom = np.dot(z, d)
    theta = np.dot(z, y) / denom
    influence = z * (y - theta * d) / denom
    _, gi = np.unique(groups, return_inverse=True)
    se = np.sqrt(np.sum(np.bincount(gi, weights=influence) ** 2))
    return float(theta), float(se)

def insulin_rdd(target, horizon=1, bandwidth=10, history=True):
    running = panel.glucose_finger.to_numpy()
    cutoffs = np.array([150., 180., 220., 260., 300.])
    nearest = np.argmin(abs(running[:, None] - (cutoffs - .5)), axis=1)
    r = running - (cutoffs[nearest] - .5)
    above = (r >= 0).astype(float)
    future = g[target].shift(-horizon).to_numpy()
    mask = (np.isfinite(running) & np.isfinite(future) &
            (abs(r) < bandwidth) & (t >= 0) & (t <= 23))
    rr, aa, kk = r[mask], above[mask], nearest[mask]
    indicators = np.eye(len(cutoffs))[kk]
    C = np.column_stack([
        indicators, indicators * rr[:, None],
        indicators * (rr * aa)[:, None],
        t[mask], (t[mask] / 24.) ** 2
    ])
    if history and horizon > 0:
        names = [c for c in controls.columns
                 if c not in ["glucose_finger", "glucose_finger_square"]]
        H = controls.loc[mask, names].to_numpy()
        H = (H - H.mean(0)) / np.maximum(H.std(0), 1e-8)
        C = np.column_stack([C, H])
    w = np.sqrt(np.maximum(1 - abs(rr) / bandwidth, 0))
    C = C * w[:, None]
    Z = aa * w
    D = panel.insulin.to_numpy()[mask] * w
    Y = future[mask] * w
    residuals = np.column_stack([Z, D, Y])
    residuals -= C @ np.linalg.lstsq(C, residuals, rcond=None)[0]
    zr, dr, yr = residuals.T
    theta, se = clustered_ratio(zr, dr, yr, sid[mask])
    first_stage = np.dot(zr, dr) / np.dot(zr, zr)
    reduced_form = np.dot(zr, yr) / np.dot(zr, zr)
    return dict(n=int(mask.sum()), rate_jump=float(first_stage),
                outcome_jump=float(reduced_form), effect=theta, se=se)

print("\n=== INSULIN THRESHOLD-INSTRUMENT ESTIMATES ===")
rdd_results = {}
for target in ["glucose_finger", "glucose_lab"]:
    for bw in [6, 10, 14]:
        result = insulin_rdd(target, 1, bw, True)
        rdd_results[(target, 1, bw)] = result
        print(target, "horizon 1, bandwidth", bw,
              {k: round(v, 5) for k, v in result.items()})
for horizon in [2, 4]:
    result = insulin_rdd("glucose_finger", horizon, 10, True)
    rdd_results[("glucose_finger", horizon, 10)] = result
    print("glucose_finger", "horizon", horizon,
          {k: round(v, 5) for k, v in result.items()})
placebo = insulin_rdd("glucose_finger", -1, 10, False)
print("Prior-reading placebo (no history adjustment):",
      {k: round(v, 5) for k, v in placebo.items()})

print("\n=== CROSS-FITTED NONLINEAR RESIDUAL ESTIMATES ===")
dml_results = {}
dml_cache = {}
start = time.time()
for target, drug in [("glucose_finger", "insulin"),
                     ("map_arterial", "norepinephrine")]:
    future = g[target].shift(-1)
    mask = future.notna() & (t >= 0) & (t < 24)
    Xdf = controls.loc[mask].copy()
    other = DRUGS[1 - DRUGS.index(drug)]
    Xdf["other_current_drug"] = panel.loc[mask, other]
    X = Xdf.to_numpy(dtype=np.float32)
    y = future[mask].to_numpy()
    d = panel.loc[mask, drug].to_numpy()
    groups = sid[mask]
    folds = (groups - 1010000000) % 2
    yp, dp = np.zeros(len(y)), np.zeros(len(y))
    for fold in [0, 1]:
        tr = np.flatnonzero(folds != fold)
        te = np.flatnonzero(folds == fold)
        if len(tr) > 140000:
            tr = np.random.default_rng(710 + fold).choice(tr, 140000, replace=False)
        for response, prediction in [(y, yp), (d, dp)]:
            nuisance = HistGradientBoostingRegressor(
                max_iter=150, learning_rate=.08, max_leaf_nodes=31,
                min_samples_leaf=75, l2_regularization=10,
                max_bins=128, random_state=90 + fold)
            nuisance.fit(X[tr], response[tr])
            prediction[te] = nuisance.predict(X[te])
    yr, dr = y - yp, d - dp
    fresh = panel.loc[mask, target].notna().to_numpy()
    for label, sub in [("all", np.ones(len(y), dtype=bool)), ("fresh", fresh)]:
        theta, se = clustered_ratio(dr[sub], dr[sub], yr[sub], groups[sub])
        result = dict(n=int(sub.sum()), effect=theta, se=se,
                      dose_residual_sd=float(dr[sub].std()),
                      outcome_residual_sd=float(yr[sub].std()))
        dml_results[(target, label)] = result
        print(target, label, {k: round(v, 5) for k, v in result.items()})
    dml_cache[target] = dict(mask=mask, y=y, d=d, yp=yp, dp=dp,
                             dr=dr, yr=yr, groups=groups, Xcolumns=list(Xdf.columns))
print("DML elapsed seconds:", round(time.time() - start, 1))

print("\n=== NOREPINEPHRINE POLICY NEAR CUFF MAP 55..85 ===")
q = panel.loc[(t >= 0) & panel.map_cuff.between(55, 85),
              ["map_cuff", "norepinephrine"]]
print(q.groupby("map_cuff").norepinephrine.agg(["size", "mean"])
      .round(4).to_string())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 4 / forwards 1–2: causal diagnostics and baselines\n"
        f"- Original probe factual NLL: {baseline_nll}; allowed increase 0.005.\n"
        f"- Original dose rollouts: {baseline_rollouts!r}\n"
        f"- Insulin threshold estimates: {rdd_results!r}\n"
        f"- Prior-reading placebo: {placebo!r}\n"
        f"- Nonlinear cross-fitted estimates: {dml_results!r}\n"
        "- These analyses use only records and pre-infusion controls; "
        "cross-fitting separates stays. Assess identification assumptions "
        "and agreement before choosing target effects.\n"
        "- Python runs: 4; forward passes: 2; no edits.\n"
    )
```
