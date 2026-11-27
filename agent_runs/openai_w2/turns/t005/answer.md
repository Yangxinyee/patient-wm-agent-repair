The original model predicts insulin x2 lowers glucose by 3.226, and norepinephrine x2 raises MAP by 1.373. Those are model responses, not established effects. I’ll now use the sharp insulin dose thresholds to test subsequent readings, with earlier readings as placebos. A dose jump affecting future—but not past—readings would support a causal effect; a null or inconsistent result would challenge the model’s response.

```python
session["python_runs"] += 1
session["forward_passes"] = 1
from scipy import linalg

hours = panel.index.get_level_values("t").to_numpy()
stay_numbers = panel.index.get_level_values("stay_id").to_numpy()
hist = panel[measurement_events].groupby(level="stay_id").ffill()
past = hist.groupby(level="stay_id").shift(1)
lag_drug = panel[drug_events].groupby(level="stay_id").shift(1).fillna(0)
next_obs = panel[measurement_events].groupby(level="stay_id").shift(-1)
glu_avg = panel[["glucose_finger", "glucose_lab"]].mean(axis=1)
map_avg = pd.concat(
    [panel.map_arterial + 3, panel.map_cuff], axis=1
).mean(axis=1)
next_glu_avg = glu_avg.groupby(level="stay_id").shift(-1)
past_glu_avg = glu_avg.groupby(level="stay_id").shift(1)
next_map_avg = map_avg.groupby(level="stay_id").shift(-1)

gf_dosing = panel.groupby("glucose_finger").insulin.agg(["mean", "count"])
jumps = gf_dosing["mean"].diff()
candidates = jumps[(jumps.index >= 140) & (jumps.index <= 270)].sort_values(ascending=False)
thresholds = []
for x, jump in candidates.items():
    if jump > .35 and all(abs(x-c) >= 20 for c in thresholds):
        thresholds.append(float(x))
    if len(thresholds) == 4:
        break
thresholds = sorted(thresholds)
print("=== INSULIN DOSE JUMPS ===")
print(candidates.head(10).round(4).to_string())
print("Selected thresholds:", thresholds)
for c in thresholds:
    print(gf_dosing.loc[c-2:c+2].round(3).to_string())

print("\n=== NOREPINEPHRINE CONTROLLER AUDIT ===")
for obs in ["map_arterial", "map_cuff"]:
    a = pd.DataFrame({
        "reading": panel[obs], "rate": panel.norepinephrine,
        "previous_rate": lag_drug.norepinephrine
    })
    a = a[(hours >= 0) & a.reading.notna()]
    a["rate_change"] = a.rate - a.previous_rate
    a["band"] = pd.cut(a.reading, np.arange(40, 96, 5))
    print(obs)
    print(a.groupby("band", observed=True).agg(
        n=("rate", "size"), rate=("rate", "mean"),
        rate_change=("rate_change", "mean")
    ).round(4).to_string())

def clustered_last_coef(X, y, clusters):
    beta = linalg.lstsq(X, y, lapack_driver="gelsy")[0]
    residual = y - X @ beta
    inverse = linalg.pinvh(X.T @ X)
    leverage = X @ inverse[:, -1]
    _, groups = np.unique(clusters, return_inverse=True)
    score = np.bincount(groups, weights=leverage * residual)
    m = len(score)
    correction = (m / max(m-1, 1)) * ((len(y)-1) / max(len(y)-X.shape[1], 1))
    se = np.sqrt(correction * np.dot(score, score))
    return float(beta[-1]), float(se)

rd_results = []
def insulin_rd(outcome, label, bandwidth, adjusted=True):
    g = panel.glucose_finger.to_numpy()
    centres = np.asarray(thresholds) - .5
    nearest = np.argmin(np.abs(g[:, None] - centres[None, :]), axis=1)
    running = g - centres[nearest]
    mask = (
        np.isfinite(g) & np.isfinite(np.asarray(outcome)) &
        (hours >= 0) & (hours < 24) & (np.abs(running) <= bandwidth)
    )
    r = running[mask] / bandwidth
    high = (r > 0).astype(float)
    th = nearest[mask]
    columns = []
    for k in range(len(thresholds)):
        indicator = (th == k).astype(float)
        columns.extend([indicator, indicator*r, indicator*r*high])
    if adjusted:
        for event in measurement_events:
            scale = 50 if "glucose" in event else 20
            x = past[event].to_numpy()[mask]
            columns.extend([
                np.nan_to_num(x, nan=float(hist[event].median())) / scale,
                np.isfinite(x).astype(float)
            ])
        for event in ["glucose_lab", "map_arterial", "map_cuff"]:
            x = panel[event].to_numpy()[mask]
            scale = 50 if "glucose" in event else 20
            columns.extend([
                np.nan_to_num(x, nan=float(hist[event].median())) / scale,
                np.isfinite(x).astype(float)
            ])
        columns.extend([
            lag_drug.insulin.to_numpy()[mask],
            lag_drug.norepinephrine.to_numpy()[mask],
            hours[mask]/24, (hours[mask]/24)**2
        ])
    columns.append(high)
    X = np.column_stack(columns)
    y = np.asarray(outcome)[mask].astype(float)
    dose = panel.insulin.to_numpy()[mask]
    clusters = stay_numbers[mask]
    first_stage, first_se = clustered_last_coef(X, dose, clusters)
    reduced, _ = clustered_last_coef(X, y, clusters)
    effect = reduced / first_stage
    _, residual_se = clustered_last_coef(X, y-effect*dose, clusters)
    se = residual_se / abs(first_stage)
    result = {
        "outcome": label, "bandwidth": bandwidth, "adjusted": adjusted,
        "n": len(y), "dose_jump": round(first_stage, 4),
        "effect_per_unit": round(effect, 4),
        "cluster_se": round(se, 4),
        "ci95": [round(effect-1.96*se, 4), round(effect+1.96*se, 4)]
    }
    rd_results.append(result)
    print(result)

print("\n=== INSULIN THRESHOLD IV/RD ESTIMATES ===")
for bandwidth in [5, 10, 14]:
    for outcome, label in [
        (next_obs.glucose_finger, "next_finger"),
        (next_glu_avg, "next_glucose_combined"),
        (past_glu_avg, "past_glucose_PLACEBO"),
        (next_obs.glucose_finger.notna().astype(float), "next_finger_observed")
    ]:
        insulin_rd(outcome, label, bandwidth)
insulin_rd(next_glu_avg, "next_glucose_combined", 5, adjusted=False)
insulin_rd(next_map_avg, "next_MAP_cross_drug_PLACEBO", 10)

print("\nCOUNTS:", session)
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Original rollout, forward 1\n"
        "100 tuning stays: insulin x2 -3.226; insulin half +1.825; "
        "NE x2 +1.373; NE half -0.846. These are predictions, not truth.\n"
        "\n## Run 4: dose-threshold identification\n"
        + json.dumps({"thresholds": thresholds, "RD": rd_results}, indent=2)
        + "\n"
    )
```