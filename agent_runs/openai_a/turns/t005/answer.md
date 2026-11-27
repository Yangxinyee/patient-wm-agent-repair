The original rollout predicts insulin doubling changes glucose by only −1.85, and predicts lower MAP for both norepinephrine edits. Those responses need checking against the records. I’ll test whether disagreements between simultaneous sensors provide independent dose variation. Strong dose relevance, little persistence, and no association with earlier outcomes would support that approach; failed checks would undermine it.

```python
print("=== ORIGINAL ROLLOUT FILE ===")
dose_baseline = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print(dose_baseline.head(4).to_string(index=False))
print("Columns:", dose_baseline.columns.tolist())

print("\n=== EXAMPLE HOURLY RECORD ===")
print(panel.loc[1010000000].head(9).round(3).to_string())

def residualize(X, a):
    return a - X @ lstsq(X, a, lapack_driver="gelsy")[0]

def sensor_iv(outcome, drug, sensor_a, sensor_b, mode="history"):
    # Outcome is the NEXT reading, not its difference from the noisy current reading.
    next_y = pgroup[outcome].shift(-1).to_numpy().astype(float)
    z = (panel[sensor_a]-panel[sensor_b]).to_numpy().astype(float)
    d = panel[drug].to_numpy().astype(float)
    good = (time_array >= 0) & (time_array < 24) & np.isfinite(z) & np.isfinite(next_y)
    ix = np.flatnonzero(good)
    keep = [j for j, name in enumerate(history_names)
            if not (name.startswith(sensor_a) or name.startswith(sensor_b))
            or "_lag" in name]
    X = np.column_stack([np.ones(len(ix)), HX[ix][:, keep]])
    if mode in ["past_difference", "stayFE"]:
        diff = panel[sensor_a]-panel[sensor_b]
        past_diff = diff.groupby(level="stay_id").ffill().groupby(level="stay_id").shift(1)
        X = np.column_stack([X, past_diff.fillna(diff.median()).to_numpy()[ix],
                             past_diff.notna().to_numpy()[ix].astype(float)])
    Y, D, Z = next_y[ix].copy(), d[ix].copy(), z[ix].copy()
    groups, ids = pd.factorize(stay_array[ix], sort=False)
    if mode == "stayFE":
        counts = np.bincount(groups)
        for A in [X, Y[:, None], D[:, None], Z[:, None]]:
            for j in range(A.shape[1]):
                A[:, j] -= (np.bincount(groups, weights=A[:, j])/counts)[groups]
    yr, dr, zr = [residualize(X, a) for a in [Y, D, Z]]
    den = zr @ dr
    beta = (zr @ yr)/den
    scores = np.bincount(groups, weights=zr*(yr-beta*dr))
    se = np.sqrt(scores@scores)/abs(den)
    fs = den/(zr@zr)
    fs_scores = np.bincount(groups, weights=zr*(dr-fs*zr))
    fs_se = np.sqrt(fs_scores@fs_scores)/(zr@zr)
    print(drug, "->", outcome, mode, "n=", len(ix),
          "IV effect=", round(beta, 4), "cluster SE=", round(se, 4),
          "first-stage=", round(fs, 5), "first-stage t=", round(fs/fs_se, 2))
    return {"beta": beta, "se": se, "first_stage": fs,
            "first_stage_t": fs/fs_se, "n": len(ix)}

print("\n=== SENSOR DISAGREEMENT DIAGNOSTICS ===")
for a, b, drug in [("glucose_finger", "glucose_lab", "insulin"),
                   ("map_cuff", "map_arterial", "norepinephrine")]:
    diff = panel[a]-panel[b]
    mean = (panel[a]+panel[b])/2
    lag_diff = diff.groupby(level="stay_id").shift(1)
    good = diff.notna() & lag_diff.notna()
    print(a, "-", b, "mean/SD:", round(diff.mean(), 3), round(diff.std(), 3),
          "corr with current mean:", round(diff.corr(mean), 3),
          "one-hour autocorrelation:", round(diff[good].corr(lag_diff[good]), 3))
    print("Difference quartiles:",
          diff.quantile([.01, .25, .5, .75, .99]).round(2).to_dict())

print("\n=== SENSOR-INSTRUMENT ESTIMATES: ASSUMPTIONS UNDER TEST ===")
iv_results = {}
for drug, a, b, outcomes in [
    ("insulin", "glucose_finger", "glucose_lab", ["glucose_finger", "glucose_lab"]),
    ("norepinephrine", "map_cuff", "map_arterial", ["map_arterial", "map_cuff"]),
]:
    for outcome in outcomes:
        for mode in ["history", "past_difference", "stayFE"]:
            iv_results[(drug, outcome, mode)] = sensor_iv(outcome, drug, a, b, mode)

with open("/work/notes.md", "a") as f:
    f.write("\n## Original rollout and identification checks; Python 4, forward 1\n")
    f.write("- Original first 100 tuning stays: insulin x2 -1.847, half +0.777; "
            "NE x2 -0.498, half -0.600.\n")
    f.write("- Sensor disagreement is only a candidate instrument; validity and relevance "
            "must be checked, not assumed.\n")
    f.write(json.dumps({str(k): v for k, v in iv_results.items()}, indent=2)+"\n")
```