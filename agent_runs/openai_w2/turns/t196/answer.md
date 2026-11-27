The new objective reduces in-sample norepinephrine variation by about 42% at matched doubled-dose sensitivity. That is promising, but it is not yet evidence of a better weight edit. I’ll cross-validate by stay, first verifying the state-to-probe alignment. The improvement must survive held-out stays and avoid substantial cross-drug responses.

```python
print("=== VERIFY STAY ALIGNMENT ===")
sid172 = np.asarray(S["stay_id"])
first172 = np.r_[True, sid172[1:] != sid172[:-1]]
last172 = np.r_[sid172[:-1] != sid172[1:], True]
matches172 = []
for label172, keep172 in [
    ("exclude_first", ~first172), ("exclude_last", ~last172)
]:
    if keep172.sum() != len(F147):
        continue
    mismatch172 = {}
    for drug172 in ["insulin", "norepinephrine"]:
        expected172 = probe_masks171[(drug172, 2.0)]
        actual172 = np.asarray(S["has_" + drug172])[keep172].astype(bool)
        mismatch172[drug172] = int(np.count_nonzero(expected172 != actual172))
    print(label172, mismatch172)
    if not any(mismatch172.values()):
        matches172.append(keep172)
assert len(matches172) == 1, "Need unambiguous metadata alignment before CV."
bundle_stays172 = sid172[matches172[0]]
fact_stays172 = np.repeat(bundle_stays172, 4)
probe_stays172 = np.concatenate([
    np.repeat(bundle_stays172[probe_masks171[k172]], 4)
    for k172 in probe_keys171
])
probe_labels172 = np.concatenate([
    np.full(len(Dparts147[j172]), j172, dtype=np.int8)
    for j172 in range(4)
])
assert len(probe_stays172) == len(D147)

unique172 = np.unique(bundle_stays172)
rng172 = np.random.default_rng(172)
shuffled172 = rng172.permutation(unique172)
fold_lookup172 = {int(s172): j172 % 5
                  for j172, s172 in enumerate(shuffled172)}
fact_folds172 = np.array([fold_lookup172[int(s172)] for s172 in fact_stays172])
probe_folds172 = np.array([fold_lookup172[int(s172)] for s172 in probe_stays172])

cv_keys172 = [(0.01, 1e-6), (0.03, 1e-6),
              (0.1, 1e-6), (0.01, 1e-4)]
cv_results172 = []
cv_predictions172 = {}
for pw172, ridge172 in cv_keys172:
    pf172 = np.zeros(len(X147))
    pp172 = np.zeros(len(D147))
    fold_gains172 = []
    for fold172 in range(5):
        tf172 = fact_folds172 != fold172
        tp172 = probe_folds172 != fold172
        Xtr172, Dtr172 = X147[tf172], D147[tp172]
        Bcv172 = fit_zero_fact171(
            Xtr172.T @ Xtr172 / len(Xtr172),
            Dtr172.T @ Dtr172 / len(Dtr172),
            Dtr172.T @ Y147[tp172] / len(Dtr172),
            pw172, ridge172
        )
        pf172[~tf172] = X147[~tf172] @ Bcv172[:, 1]
        pp172[~tp172] = D147[~tp172] @ Bcv172[:, 1]
        m172 = (~tp172) & (probe_labels172 == 3)
        y172 = Y147[m172, 1]
        fold_gains172.append(float(pp172[m172] @ y172 / (y172 @ y172)))

    metrics172 = {}
    for j172, label172 in [(2, "half"), (3, "x2")]:
        m172 = probe_labels172 == j172
        y172 = Y147[m172, 1]
        metrics172[label172 + "_gain"] = float(
            pp172[m172] @ y172 / (y172 @ y172)
        )
    rms172 = float(np.sqrt(np.mean(pf172 ** 2)))
    insmask172 = probe_labels172 < 2
    insy172 = Y147[insmask172, 0]
    cross172 = float(pp172[insmask172] @ insy172 / (insy172 @ insy172))
    cv_results172.append({
        "probe_weight": pw172, "ridge": ridge172, "cv_ne_rms": rms172,
        **metrics172,
        "matched_cv_rms": rms172 * reference171["ne_x2_gain"]
                          / metrics172["x2_gain"],
        "fold_x2_min": min(fold_gains172),
        "fold_x2_max": max(fold_gains172),
        "cross_NE_per_insulin": cross172
    })
    cv_predictions172[(pw172, ridge172)] = (pf172, pp172)

print("\n=== STAY-CROSSFITTED RESULTS ===")
print(pd.DataFrame(cv_results172).to_string(
    index=False, float_format=lambda x: f"{x:.6g}"))
print("Validated factual NE RMS:", reference171["ne_rms"])

current_probe172 = (D147[:, 1:] * sd147) @ validated_W151[:, 1]
insmask172 = probe_labels172 < 2
insy172 = Y147[insmask172, 0]
print("Validated cross-NE response per insulin unit:",
      float(current_probe172[insmask172] @ insy172 / (insy172 @ insy172)))
print("Weights unchanged:",
      np.array_equal(load("core.b_in.weight"), validated_B151))
```
