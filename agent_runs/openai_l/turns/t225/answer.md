The records support delayed insulin effects and faster-decaying norepinephrine effects; I’ll retain those estimates rather than the rejected state-space fits. The head has 192 GELU units, so a few may be reusable for sensor-gated dose features. First I’ll test family-specific dose decoders and verify the target conventions. Lower factual noise with preserved dose contrasts would support this approach.

```python
print("=== SENSOR / SPLIT DEFINITIONS ===")
print("events:", np.unique(events179, return_counts=True))
print("codes:", VOCAB["codes"])
print("fit/validation readings:", int(fitmask179.sum()), int(valmask179.sum()))
for ev in np.unique(events179):
    gg = X200[:, events179 == ev] @ gatecanon200
    print("gate", ev, "mean", float(gg.mean()), "SD", float(gg.std()),
          "range", np.quantile(gg, [.001, .999]).round(4))
print("quiet units:", quiet192)
print("raw input magnitude:", float(np.std(X200[..., :192])))

print("\n=== CANONICAL CASE STRUCTURE ===")
for key in (("insulin", 2.0), ("norepinephrine", 2.0)):
    value = canon_cases200[key]
    print(key, type(value).__name__, len(value))
    for j, item in enumerate(value):
        print(" item", j, type(item).__name__,
              tuple(item.shape) if hasattr(item, "shape") else repr(item)[:350])
print("\n=== PAIRED CASE STRUCTURE ===")
for drug, value in pair179.items():
    print(drug, "length", len(value))
    for item in value[:2]:
        if isinstance(item, tuple):
            print([(type(z).__name__, tuple(z.shape) if hasattr(z, "shape")
                    else list(z)[:12] if isinstance(z, dict) else repr(z)[:100])
                   for z in item])
        elif isinstance(item, dict):
            print({k: tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
                   for k, v in item.items()})
        else:
            print(type(item).__name__, repr(item)[:350])

print("\n=== FACTUAL LOGIT ANCHOR ===")
print("Lorig-Lquiet RMS:", float(np.sqrt(np.mean((Lorig179-L179)**2))))
print("Yfact versus difference RMS:",
      float(np.sqrt(np.mean((Yfact179-(Lorig179-L179))**2))))
print("\n=== RELEVANT LIVE HELPERS / ROUTES ===")
helper203 = [k for k in list(globals()) if callable(globals()[k]) and
             any(s in k.lower() for s in
                 ("step", "encode", "state", "feature", "decoder", "flow", "head"))]
print(" ".join(sorted(helper203)))
for k in ("B167", "B160", "R118", "U167", "wire_out152", "Wrepair125"):
    if k in globals():
        v = globals()[k]
        print(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
print("\nNo weights changed.")
```