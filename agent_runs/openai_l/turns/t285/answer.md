Most of this factual audit’s penalty comes from glucose_lab and map_cuff, while the primary sensors are nearly unchanged. I’ll check whether systematic sensor-specific reconstruction errors explain that cost. If small analytical offsets restore factual logits while preserving dose contrasts, they could free room for a stronger repair.

```python
print("=== FACTUAL / CAUSAL GRID SCORES ===")
for key252, val252 in expanded234.items():
    if key252[2:] in [(0.0, 0.0), (0.3, 0.3), (1.0, 1.0)]:
        print(key252, {k: val252[k] for k in
              ["train_delta", "val_delta", "tune_delta", "slopes"]
              if k in val252})
for key252, val252 in grid242.items():
    print("grid", key252, val252["score"])

print("=== BROAD FIT STATISTICS ===")
for key252, val252 in stats234.items():
    print(key252, [(k, type(v).__name__, getattr(v, "shape", ""))
                   for k, v in val252.items()])

print("=== LABEL / REGISTER CACHE OBJECTS ===")
for name252, val252 in sorted(list(globals().items())):
    if re.search(r"(179|204|214|225|228|234)$", name252):
        if not inspect.isfunction(val252):
            print(name252, type(val252).__name__,
                  getattr(val252, "shape", ""),
                  list(val252.keys())[:6] if isinstance(val252, dict) else "")

Tlive252 = np.vstack([
    np.asarray(load("attr_heads.value_bin.2.weight")).T,
    np.asarray(load("attr_heads.value_bin.2.bias"))[None, :]
])
print("=== LIVE FINAL HEAD CHECK ===",
      np.max(np.abs(Tlive252 - grid242[(.5, .4)]["T"])))
print("=== TRAIN LOGIT RECONSTRUCTION ERROR BY SENSOR ===")
residual252 = Lorig179 - (arch235["F"] @ Tlive252 + offset204)
residual252 -= residual252.mean(axis=-1, keepdims=True)
for event252 in np.unique(events179):
    use252 = np.asarray(events179) == event252
    rr252 = residual252[:, use252]
    print(event252, "n", int(use252.sum()),
          "RMS", float(np.sqrt(np.mean(rr252**2))),
          "mean-error RMS", float(np.sqrt(np.mean(rr252.mean(axis=(0, 1))**2))),
          "mean error", np.round(rr252.mean(axis=(0, 1)), 4))
print("nll179 interface:", inspect.signature(nll179))
```