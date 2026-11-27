The asymmetry is already present at hour 13, so later feedback cannot be its only cause. I’ll inspect the paired hidden-unit activations and calibration parameters. A nonlinear response with history held fixed would support changing the decoder; otherwise the dose-transport path needs investigation.

```python
print("=== INSTALLED CALIBRATION ===")
print("strengths:", _r149_strengths)
print("reserved neurons:", _r148_reserved.tolist())
print("head tensor names:", _r149_names)
for j153, spec153 in enumerate(_r148_specs):
    print("\nSPEC", j153)
    for key153, val153 in spec153.items():
        if isinstance(val153, (np.ndarray, torch.Tensor)):
            ar153 = np.asarray(val153)
            print(key153, "shape", ar153.shape,
                  "range", (float(ar153.min()), float(ar153.max())),
                  "norm", float(np.linalg.norm(ar153)))
        elif hasattr(val153, "coef_"):
            print(key153, type(val153).__name__,
                  "coef shape", np.asarray(val153.coef_).shape,
                  "intercept", np.asarray(val153.intercept_).tolist())
        else:
            print(key153, repr(val153)[:500])

print("\n=== RESERVED-NEURON ACTIVATIONS BY SENSOR ===")
pre153 = (
    _r147_wc.astype(np.float64)
    @ load("attr_heads.value_bin.0.weight").astype(np.float64).T
    + load("attr_heads.value_bin.0.bias").astype(np.float64)
)
for event153 in _r148_events:
    mask153 = _r148_event_array == event153
    print(event153, "n", int(mask153.sum()))
    vals153 = pre153[:, mask153][:, :, _r148_reserved].reshape(-1, 8)
    print("quantiles 10/50/90 by reserved neuron:")
    print(np.round(np.quantile(vals153, [.1, .5, .9], axis=0), 3))

print("\n=== RECORD IDENTIFICATION CHECKS ===")
notes153 = Path("/work/notes.md").read_text()
print(notes153[4500:7400])
```
