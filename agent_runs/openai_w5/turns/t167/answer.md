The reserved dose coordinates have equalized normalization and very fast decay, suggesting an intended one-step exposure channel rather than lasting drug memory. I’ll check its actual rate decoding, the sensor gates, and the fitted corrections next. Accurate dose recovery and clean sensor separation would support installing a candidate; leakage or nonlinear decoding would not.

```python
print("=== CURRENT CALIBRATION ARRAYS ===")
for prefix145 in ("_r139_", "_r140_", "_r141_"):
    for n145, o145 in list(globals().items()):
        if not n145.startswith(prefix145):
            continue
        if hasattr(o145, "shape"):
            print(n145, type(o145).__name__, tuple(o145.shape))
        elif isinstance(o145, dict) and len(o145) < 25:
            print(n145, "dict", list(o145))
        elif isinstance(o145, (list, tuple)):
            print(n145, "sequence", len(o145))
        elif isinstance(o145, (int, float, str)):
            print(n145, repr(o145)[:100])

print("=== SELECTED FIT CONTENTS ===")
for n145 in (
    "_r141_best", "_r141_nuisances", "_r141_tilts",
    "_r140_gate", "_r140_gate_score", "_r140_signed",
):
    o145 = globals().get(n145)
    if isinstance(o145, dict):
        for k145, v145 in o145.items():
            if hasattr(v145, "coef_"):
                print(n145, k145, type(v145).__name__,
                      "coef shape", v145.coef_.shape,
                      "intercept", np.asarray(v145.intercept_))
            elif isinstance(v145, dict):
                print(n145, k145, {
                    kk: (tuple(vv.shape) if hasattr(vv, "shape") else repr(vv)[:100])
                    for kk, vv in v145.items()
                })
            else:
                print(n145, k145, repr(v145)[:240])
    elif hasattr(o145, "shape"):
        a145 = np.asarray(o145)
        print(n145, "range", np.nanmin(a145), np.nanmax(a145),
              "quantiles", np.nanquantile(a145, [0, .05, .5, .95, 1]))
    else:
        print(n145, repr(o145)[:600])

print("=== CALIBRATION SAMPLE ===")
print(_r140_df.head(3).to_string(index=False))
print("Stays:", len(np.unique(_r140_stays)),
      "fit/test rows:", np.sum(_r140_fitmask), np.sum(_r140_testmask))
print("Dose quantiles:", np.quantile(_r140_doses, [0, .5, .9, .99, 1], axis=0))
print("Pair-coordinate quantiles:", np.quantile(_r140_pair, [0, .5, .9, .99, 1], axis=0))
print("Proxy shape:", np.shape(_r141_proxy))
print("Vocab keys:", list(VOCAB))
print("Codes:", VOCAB["codes"])
print("Value centres:", VOCAB["value_bin_centres"])
print("W0 equals installed head:",
      torch.equal(W0["attr_heads.value_bin.0.weight"],
                  load("attr_heads.value_bin.0.weight")))
```
