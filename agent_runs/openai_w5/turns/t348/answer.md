The pooled increase is +0.004145 nats, but its bootstrap upper bound exceeds the limit. The edit also raises factual glucose predictions by about 1.15 on average. I’ll inspect the sensor-specific readout neurons for a small calibration correction; it must improve loss without weakening the dose responses.

```python
print("VOCAB STRUCTURE")
print("Keys:", list(VOCAB))
print("Codes:", repr(VOCAB["codes"])[:1800])
for key in VOCAB:
    if "edge" in key or "centre" in key:
        print(key, repr(VOCAB[key])[:3500])

print("\nSENSOR NEURONS")
print(sensor_neurons)
W0294 = array293(load("attr_heads.value_bin.0.weight"))
B0294 = array293(load("attr_heads.value_bin.0.bias"))
W2294 = array293(load("attr_heads.value_bin.2.weight"))
for event, pair in sensor_neurons.items():
    ids = list(pair)
    print(event, "neurons", ids, "first-layer biases", B0294[ids])
    print("Output columns:\n", np.round(W2294[:, ids], 5))
    print("Private input coefficients:\n",
          np.round(W0294[np.ix_(ids, [21, 61, 15, 66])], 6))

print("\nCENTER FIT METADATA")
for event, fit in center_fits275.items():
    print(event, "keys", list(fit))
    for key, value in fit.items():
        if np.isscalar(value):
            print(key, value)

print("\nFACTUAL BIN-CALIBRATION DATA")
for event, d in audit293.groupby("event"):
    print(event, "n", len(d),
          "observed range", (d.observed.min(), d.observed.max()),
          "edited residual second moment",
          float(np.mean((d.predicted_mean - d.observed) ** 2)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 294: sensor-calibration preparation\n"
            "- Current factual glucose predictions have an upward shift, but "
            "a mean correction alone is not yet known to repair their NLL.\n"
            "- Inspecting reserved signal/control neurons and per-code fine-bin "
            "geometry for a small, non-gradient intercept calibration.\n"
            "- No weights changed. Any calibration requires paired factual and "
            "finite-rollout validation.\n")
```