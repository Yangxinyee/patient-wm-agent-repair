The coefficient conversion did not reproduce the reported factual residual scale, so I will not install it. This may be a layout issue rather than a failed regression. I’ll test the intercept convention against the saved predictions and check the selected bundle indices; agreement would allow the local sensitivity test to proceed.

```python
print("=== DESIGN CONVENTION CHECK ===")
for fam118, arr118 in [("E", prelude_E111), ("ffn", prelude_hidden111)]:
    fit118 = prelude_fits112[fam118][0.3]
    zz118 = (arr118.astype(np.float64) - fit118["mu"]) / fit118["sd"]
    B118 = fit118["B"]
    pred_last118 = zz118 @ B118[:-1] + B118[-1]
    pred_first118 = zz118 @ B118[1:] + B118[0]
    print(fam118, "reported", fit118["report"])
    print("intercept last: RMS", np.sqrt(np.mean(pred_last118**2, axis=(0,1))),
          "means", pred_last118.mean(axis=(0,1)))
    print("intercept first: RMS", np.sqrt(np.mean(pred_first118**2, axis=(0,1))),
          "means", pred_first118.mean(axis=(0,1)))
    print("sd range:", float(fit118["sd"].min()), float(fit118["sd"].max()))

print("\n=== SAVED DESIGN SAMPLES ===")
print("dxt112 first row:", dxt112[0, :5], "...", dxt112[0, -5:])
print("target112 first 5:", target112[:5])
print("dose_actual first 5:", dose_actual_repair[:5])
print("Saved 112 variable layouts:")
print([(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
       for k, v in list(globals().items()) if k.endswith("112")])

print("\n=== CORE TEST ALIGNMENT ===")
for k in sorted(list(globals())):
    if k.endswith("104") and not callable(globals()[k]):
        v = globals()[k]
        print(k, describe115(v))
print("core_rows104 first:", np.asarray(core_rows104)[:6])
print("repair_start_rows101 first:", repair_start_rows101[:4])
print("S keys:", list(S))
print("S x shape:", S["x"].shape)
print("S equals original archive:",
      np.max(np.abs(S["x"] - state_array110)))
```
