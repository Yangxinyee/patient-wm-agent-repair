Nonlinear adjustment leaves insulin near −2 per unit/hour across both sensors. Norepinephrine is near +3 after startup, but its hour-0 estimates remain inconsistent. I’ll check sharp changes in the recorded dosing policy and pre-dose placebo outcomes. A dose jump followed by an outcome jump—but no pre-dose jump—would provide stronger evidence than regression alone.

```python
RUN_COUNTS["python"] += 1

print("=== SIMULTANEOUS SENSOR DIFFERENCES ===")
for first, second in [MEAS[:2], MEAS[2:]]:
    paired = panel[[first, second]].dropna()
    diff = paired[first] - paired[second]
    print(first, "-", second, "n=", len(diff),
          "mean=", round(float(diff.mean()), 4),
          "SD=", round(float(diff.std()), 4))

print("\n=== INSULIN POLICY NEAR POSSIBLE THRESHOLDS, HOUR 0 ===")
p0 = panel.xs(0, level="t")
policy0 = p0.loc[p0.glucose_finger.notna()].groupby("glucose_finger").agg(
    n=("insulin", "size"), mean_rate=("insulin", "mean")
)
cuts = []
for low, high in [(145, 155), (175, 185), (215, 225)]:
    tab = policy0.reindex(np.arange(low, high + 1))
    print(tab.round(3).to_string())
    jump = tab.mean_rate.diff()
    jump.loc[(tab.n < 20) | (tab.n.shift(1) < 20)] = np.nan
    cut = float(jump.idxmax()) - 0.5
    cuts.append(cut)
print("Empirically selected integer-reading boundaries:", cuts)

print("\n=== NE POLICY BY ARTERIAL MAP, HOUR 0 ===")
ne_policy = p0.loc[p0.map_arterial.notna()].groupby("map_arterial").agg(
    n=("norepinephrine", "size"),
    mean_rate=("norepinephrine", "mean"),
    active=("norepinephrine", lambda v: (v > 0).mean())
)
print(ne_policy.loc[(ne_policy.index >= 55) & (ne_policy.index <= 75)]
      .round(3).to_string())

# Local linear fuzzy RD: threshold crossing instruments the dose.
# Discrete readings and the smoothness assumption remain limitations.
rd_control_names = [
    "t", "t2", "glucose_finger_lag1", "glucose_finger_lag2",
    "glucose_finger_baseline", "insulin_lag1", "insulin_lag2",
    "norepinephrine_lag1", "map_cuff_last"
]
reading = panel.glucose_finger.to_numpy()
distances = reading[:, None] - np.asarray(cuts)[None, :]
nearest = np.argmin(np.nan_to_num(np.abs(distances), nan=np.inf), axis=1)
distance = distances[np.arange(len(panel)), nearest]
above = (distance > 0).astype(float)

def rd_estimate(y, bandwidth, only_hour0=False):
    mask = (valid_time & np.isfinite(reading) & np.isfinite(y)
            & (np.abs(distance) <= bandwidth))
    if only_hour0:
        mask &= tt == 0
    r = distance[mask]
    z = above[mask]
    which = nearest[mask]
    onehot = np.eye(len(cuts))[which]
    c = np.column_stack([
        onehot, onehot * r[:, None], onehot * (r * z)[:, None],
        HC.loc[mask, rd_control_names].to_numpy()
    ]).astype(np.float64)
    scales = np.sqrt(np.mean(c**2, axis=0))
    c = c[:, scales > 1e-10] / scales[scales > 1e-10]
    w = 1 - np.abs(r) / (bandwidth + 0.5)
    sw = np.sqrt(w)
    cw = c * sw[:, None]
    zr = z - c @ linalg.lstsq(cw, z * sw)[0]
    a = panel.insulin.to_numpy()[mask]
    yy = np.asarray(y)[mask]
    denom = float(np.dot(w * zr, a))
    first_stage = denom / float(np.dot(w * zr, z))
    reduced_form = float(np.dot(w * zr, yy)) / float(np.dot(w * zr, z))
    beta = reduced_form / first_stage
    structural_residual = yy - beta * a
    structural_residual -= c @ linalg.lstsq(
        cw, structural_residual * sw
    )[0]
    _, gi = np.unique(stay_array[mask], return_inverse=True)
    score = np.bincount(gi, weights=w * zr * structural_residual)
    se = float(np.sqrt(np.sum(score**2)) / abs(denom))
    return {
        "n": int(mask.sum()), "dose_jump": first_stage,
        "outcome_jump": reduced_form, "IV_effect": beta, "cluster_se": se
    }

rd_results = []
for sensor in ("glucose_finger", "glucose_lab"):
    y = panel[sensor].groupby(level="stay_id").shift(-1).to_numpy()
    for bw in (6, 10, 15):
        for initial_only in (False, True):
            ans = rd_estimate(y, bw, initial_only)
            ans.update(sensor=sensor, bandwidth=bw,
                       sample="hour 0" if initial_only else "all hours")
            rd_results.append(ans)

# This lab reading precedes the dose, so the dose cannot cause its jump.
for bw in (6, 10, 15):
    ans = rd_estimate(panel.glucose_lab.to_numpy(), bw, False)
    ans.update(sensor="PRE-DOSE lab placebo", bandwidth=bw, sample="all hours")
    rd_results.append(ans)

rd_table = pd.DataFrame(rd_results)
print("\n=== LOCAL DOSE-POLICY DISCONTINUITY CHECK ===")
print(rd_table[["sensor", "sample", "bandwidth", "n", "dose_jump",
                "outcome_jump", "IV_effect", "cluster_se"]]
      .round(4).to_string(index=False))

print("\n=== DOSE-INNOVATION EFFECTS BY PREDICTED DOSE ===")
heterogeneity = []
for drug, sensors, threshold in [
    ("insulin", MEAS[:2], 1.5),
    ("norepinephrine", MEAS[2:], 0.5)
]:
    ra = dose_residuals[drug]
    predicted_a = panel[drug].to_numpy() - ra
    for sensor in sensors:
        ry = outcome_residuals[sensor]
        for label, state_mask in [
            ("lower predicted dose", predicted_a < threshold),
            ("higher predicted dose", predicted_a >= threshold)
        ]:
            mask = valid_time & np.isfinite(ra) & np.isfinite(ry) & state_mask
            beta, se = innovation_slope(ra[mask], ry[mask], stay_array[mask])
            heterogeneity.append({
                "sensor": sensor, "state": label, "n": int(mask.sum()),
                "beta": beta, "cluster_se": se
            })
print(pd.DataFrame(heterogeneity).round(4).to_string(index=False))

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Identification checks — Python 5; forward 2\n"
        "- Crossfit estimates support insulin approximately -2 per unit/hour "
        "on both glucose sensors. NE approximately +3 after startup; "
        "hour-0 disagreement remains unresolved.\n"
        "- Checked empirically located insulin dose-policy discontinuities "
        "with local linear IV and a pre-dose lab placebo. Discrete running "
        "variables and smoothness assumptions limit RD certainty.\n"
        + rd_table.to_string(index=False) + "\n"
        + pd.DataFrame(heterogeneity).to_string(index=False) + "\n"
    )
print("\nCounts:", RUN_COUNTS)
```
