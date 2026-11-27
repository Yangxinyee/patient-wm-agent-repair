Adding the independent states lowers the fitted NE residual there from 0.0294 to about 0.0264, with little change on the original cohort. That is promising but is not held-out evidence. I’ll cross-validate by stay; an improvement on held-out states without losing dose sensitivity would justify a forward test.

```python
new_stays203 = np.asarray(S187["stay_id"])[r196_rows]
assert len(new_stays203) == len(E202)
unique_new203 = np.unique(new_stays203)
rng203 = np.random.default_rng(203)
shuffled203 = rng203.permutation(unique_new203)
fold_map203 = {s: j % 5 for j, s in enumerate(shuffled203)}
new_folds203 = np.repeat(
    np.array([fold_map203[s] for s in new_stays203]), 4
)
assert len(new_folds203) == len(X202)
assert len(fact_folds172) == len(X147)
assert len(probe_folds172) == len(D147)

cv_keys203 = [
    (0.0003, 1e-7), (0.003, 1e-7), (0.003, 1e-6),
    (0.01, 1e-6), (0.03, 1e-6)
]
cv_predictions203 = {
    key: {"old": np.zeros(len(X147)),
          "new": np.zeros(len(X202)),
          "probe": np.zeros(len(D147))}
    for key in cv_keys203
}
for fold203 in range(5):
    old_train203 = fact_folds172 != fold203
    new_train203 = new_folds203 != fold203
    probe_train203 = probe_folds172 != fold203
    xt203 = X147[old_train203]
    xn203 = X202[new_train203]
    dp203 = D147[probe_train203]
    yp203 = Y147[probe_train203]
    hf203 = (xt203.T @ xt203 + xn203.T @ xn203) / (len(xt203)+len(xn203))
    hp203 = dp203.T @ dp203 / len(dp203)
    gp203 = dp203.T @ yp203 / len(dp203)
    ne_train203 = ne_double202 & probe_train203
    reference203 = float(np.mean((D147 @ Bcurrent202)[ne_train203,1]))
    for key203 in cv_keys203:
        b203 = fit_zero_fact171(hf203, hp203, gp203, *key203)
        sensitivity203 = float(np.mean((D147 @ b203)[ne_train203,1]))
        b203[:,1] *= reference203 / sensitivity203
        cv_predictions203[key203]["old"][~old_train203] = (
            X147[~old_train203] @ b203[:,1])
        cv_predictions203[key203]["new"][~new_train203] = (
            X202[~new_train203] @ b203[:,1])
        cv_predictions203[key203]["probe"][~probe_train203] = (
            D147[~probe_train203] @ b203[:,1])

cv_rows203 = []
for key203, predictions203 in cv_predictions203.items():
    p203 = predictions203["probe"]
    cv_rows203.append({
        "probe_weight": key203[0], "ridge": key203[1],
        "old_CV_RMS": rms202(predictions203["old"]),
        "new_CV_RMS": rms202(predictions203["new"]),
        "pooled_CV_RMS": rms202(np.concatenate([
            predictions203["old"], predictions203["new"]])),
        "NE_double_mean": float(p203[ne_double202].mean()),
        "NE_half_mean": float(p203[ne_half202].mean())
    })
cv_table203 = pd.DataFrame(cv_rows203)
print("STAY-HELD-OUT CROSS-VALIDATION")
print(cv_table203.to_string(index=False, float_format=lambda x:f"{x:.6g}"))
print("CURRENT FIXED DECODER OLD/NEW/POOLED RMS",
      rms202((X147 @ Bcurrent202)[:,1]),
      rms202((X202 @ Bcurrent202)[:,1]),
      rms202(np.concatenate([(X147 @ Bcurrent202)[:,1],
                            (X202 @ Bcurrent202)[:,1]])))
print("CURRENT NE DOUBLE/HALF PROBE",
      float((D147 @ Bcurrent202)[ne_double202,1].mean()),
      float((D147 @ Bcurrent202)[ne_half202,1].mean()))

print("\nINDEPENDENT RECONSTRUCTION ARRAYS")
for k in sorted(list(globals())):
    v = globals()[k]
    if k.startswith(("r18", "r19")) and hasattr(v, "shape"):
        if tuple(v.shape) in [(4044,192), (4044,32), (4144,192),
                             (4044,4,192)]:
            print(k, tuple(v.shape))
print("\nRELEVANT ENCODING HELPERS")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and any(s in k.lower() for s in
                          ["atom", "encode", "batched_set", "posterior", "cf_action"]):
        try:
            print(k, inspect.signature(v))
        except (ValueError, TypeError):
            pass

with open("/work/notes.md", "a") as f:
    f.write("\n## Run203: stay-held-out decoder comparison\n"
            + cv_table203.to_string(index=False)
            + "\nWeights remain validated_B176. Decoder residuals are only "
              "a proxy for factual disturbance, not a substitute for NLL.\n")
```
