The recovered analyses support insulin lowering glucose and norepinephrine raising MAP. The fixed-dose estimates are about −10.5/+5.3 for insulin and +3.8/−1.9 for norepinephrine on the first 100 tuning stays, with greater identification uncertainty for norepinephrine. I’ll check whether factual calibration can create room for a stronger MAP repair; improvement must hold on independent tuning states.

```python
print("=== MAP FRONTIER ===")
print(map164_frontier.to_string(index=False, float_format=lambda x: f"{x:.5f}")[:3600])

print("\n=== RECONSTRUCTION FIELDS ===")
for nm169 in ["wm_train145", "wm_tune145"]:
    mc169 = globals()[nm169]
    print(nm169, {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                  for k, v in mc169.items() if k != "raw"})
print("train meta columns:", list(rebuilt_train145.columns))
print("tuning meta columns:", list(wm_tune_meta145.columns))
print("bin-axis:", wm_v151)

for nm169 in ["wm_rawloss152", "map164_mean"]:
    try:
        print("\n", nm169, inspect.getsource(globals()[nm169])[:1800])
    except (OSError, TypeError):
        print(nm169, "source unavailable; constants:", globals()[nm169].__code__.co_consts)

print("\n=== ORIGINAL SENSOR CALIBRATION ===")
for label169, mc169, meta169 in [
    ("train", wm_train145, rebuilt_train145),
    ("tuning", wm_tune145, wm_tune_meta145)
]:
    if "value" in meta169.columns and "mean" in mc169:
        mu169 = mc169["mean"]
        if hasattr(mu169, "numpy"):
            mu169 = mu169.numpy()
        err169 = meta169["value"].to_numpy() - np.asarray(mu169)
        print(label169, pd.DataFrame({"event": meta169["event"].to_numpy(),
                                      "error": err169}).groupby("event")["error"]
              .agg(["count", "mean", "std"]).to_string())

print("\n=== SENSOR COEFFICIENT NORMS ===")
for sensor169 in ["map_arterial", "map_cuff"]:
    for fw169 in [0.03, 0.3, 3.0]:
        w169 = sensor165_fits[(sensor169, fw169, 1.0)]
        print(sensor169, fw169, "norm", np.linalg.norm(w169[:192]),
              "intercept", w169[-1])
```
