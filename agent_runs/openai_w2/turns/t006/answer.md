The threshold test found an apparent insulin effect, but also a large jump in the chance of a subsequent finger reading. That recording-policy change could bias the effect estimate. I’ll test subsequent lab readings separately and check their availability around the thresholds. If lab availability stays smooth and the effect disappears, the apparent finger-reading response is likely selection rather than a drug effect.

```python
session["python_runs"] += 1

print("=== OBSERVATION FRACTIONS BY HOUR ===")
print(panel[measurement_events].notna().groupby(level="t").mean().round(3).to_string())

print("\n=== INSULIN DOSE VARIATION AT FIXED GLUCOSE ===")
for g in [155, 170, 190, 210, 230]:
    rates = panel.loc[panel.glucose_finger == g, "insulin"]
    print("finger =", g, "n =", len(rates),
          "rate frequencies =", rates.value_counts(normalize=True).head(8).round(3).to_dict())

print("\n=== EXAMPLE NOREPINEPHRINE HISTORY ===")
ne_counts = (panel.norepinephrine > 0).groupby(level="stay_id").sum()
example_id = ne_counts[ne_counts >= 12].index[0]
print("stay:", example_id)
print(panel.loc[example_id].iloc[1:15].round(3).to_string())

true_cuts = np.array([150., 180., 220., 260.])
selection_rd = []

def clean_insulin_rd(outcome, label, bandwidth, adjusted=True):
    g = panel.glucose_finger.to_numpy()
    nearest = np.argmin(np.abs(g[:, None] - true_cuts[None, :]), axis=1)
    running = g - true_cuts[nearest]
    y_all = np.asarray(outcome, dtype=float)
    mask = (
        np.isfinite(g) & np.isfinite(y_all) &
        (hours >= 0) & (hours < 24) &
        (np.abs(running) <= bandwidth) & (running != 0)
    )
    r = running[mask] / bandwidth
    high = (r > 0).astype(float)
    th = nearest[mask]
    columns = []
    for k in range(len(true_cuts)):
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
    y = y_all[mask]
    dose = panel.insulin.to_numpy()[mask]
    clusters = stay_numbers[mask]
    first_stage, _ = clustered_last_coef(X, dose, clusters)
    reduced, _ = clustered_last_coef(X, y, clusters)
    effect = reduced / first_stage
    _, residual_se = clustered_last_coef(X, y-effect*dose, clusters)
    se = residual_se / abs(first_stage)
    result = {
        "outcome": label, "bandwidth": bandwidth, "adjusted": adjusted,
        "n": len(y), "dose_jump": round(first_stage, 4),
        "effect_per_unit": round(effect, 4), "cluster_se": round(se, 4),
        "ci95": [round(effect-1.96*se, 4), round(effect+1.96*se, 4)]
    }
    selection_rd.append(result)
    print(result)

print("\n=== RD EXCLUDING ROUNDED THRESHOLD VALUES ===")
for bw in [5, 10, 14]:
    clean_insulin_rd(next_obs.glucose_lab, "next_LAB", bw)
    clean_insulin_rd(next_obs.glucose_finger, "next_FINGER", bw)

for outcome, label in [
    (next_obs.glucose_lab.notna().astype(float), "next_LAB_observed"),
    (next_obs.glucose_finger.notna().astype(float), "next_FINGER_observed"),
    (next_obs.map_cuff, "next_CUFF_cross_drug"),
    (next_obs.map_cuff.notna().astype(float), "next_CUFF_observed")
]:
    clean_insulin_rd(outcome, label, 10)

past_lab = panel.glucose_lab.groupby(level="stay_id").shift(1)
clean_insulin_rd(past_lab, "past_LAB_unadjusted_PLACEBO", 10, adjusted=False)
clean_insulin_rd(past_glu_avg, "past_GLUCOSE_unadjusted_PLACEBO", 10, adjusted=False)

print("\nCOUNTS:", session)
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 5: recording-policy falsification\n"
        "Initial insulin RD was accompanied by a large discontinuity in future "
        "finger-reading availability, threatening identification. Rounded "
        "threshold values have half-jumps, so this audit uses cuts at "
        "150/180/220/260 and excludes exact cut values. Testing lab outcomes "
        "and their availability separately, and unadjusted past-outcome "
        "placebos rather than treating history-adjusted placebos as decisive.\n"
        + json.dumps(selection_rd, indent=2) + "\n"
    )
```