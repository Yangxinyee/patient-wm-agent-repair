The inspection stopped because it treated a module’s `shape` function as an array shape. No weights changed, and it provides no new evidence about the repair. I’ll fix that check and complete the dose-decoding and sensor-gating audit before installing anything.

```python
import types
print("=== CURRENT CALIBRATION ARRAYS ===")
for prefix146 in ("_r140_", "_r141_"):
    for n146, o146 in list(globals().items()):
        if not n146.startswith(prefix146) or isinstance(o146, types.ModuleType):
            continue
        shape146 = getattr(o146, "shape", None)
        if isinstance(shape146, (tuple, torch.Size)):
            print(n146, type(o146).__name__, tuple(shape146))
        elif isinstance(o146, dict) and len(o146) < 25:
            print(n146, "dict", list(o146))
        elif isinstance(o146, (list, tuple)):
            print(n146, "sequence", len(o146))
        elif isinstance(o146, (int, float, str)):
            print(n146, repr(o146)[:100])

print("=== SELECTED FIT CONTENTS ===")
for n146 in (
    "_r141_best", "_r141_nuisances", "_r141_tilts",
    "_r140_gate", "_r140_gate_score", "_r140_signed",
):
    o146 = globals().get(n146)
    if isinstance(o146, dict):
        for k146, v146 in o146.items():
            if hasattr(v146, "coef_"):
                print(n146, k146, type(v146).__name__,
                      "coef shape", v146.coef_.shape,
                      "intercept", np.asarray(v146.intercept_))
            elif isinstance(v146, dict):
                print(n146, k146, {
                    kk: (tuple(vv.shape) if isinstance(getattr(vv, "shape", None), (tuple, torch.Size))
                         else repr(vv)[:100])
                    for kk, vv in v146.items()
                })
            else:
                print(n146, k146, repr(v146)[:240])
    elif isinstance(getattr(o146, "shape", None), (tuple, torch.Size)):
        a146 = np.asarray(o146)
        print(n146, "range", np.nanmin(a146), np.nanmax(a146),
              "quantiles", np.nanquantile(a146, [0, .05, .5, .95, 1]))
    else:
        print(n146, repr(o146)[:600])

print("=== CALIBRATION SAMPLE ===")
print(_r140_df.head(3).to_string(index=False))
print("Stays:", len(np.unique(_r140_stays)),
      "fit/test rows:", np.sum(_r140_fitmask), np.sum(_r140_testmask))
print("Dose quantiles:", np.quantile(_r140_doses, [0, .5, .9, .99, 1], axis=0))
print("Pair quantiles:", np.quantile(_r140_pair, [0, .5, .9, .99, 1], axis=0))
print("Proxy shape:", np.shape(_r141_proxy))
print("Codes:", VOCAB["codes"])
print("Centre-table shape:", np.shape(VOCAB["value_bin_centres"]))
print("W0 equals installed head:",
      torch.equal(W0["attr_heads.value_bin.0.weight"],
                  load("attr_heads.value_bin.0.weight")))
with open("/work/notes.md", "a") as f:
    f.write("\n- Run 145 inspection failed on a module's shape attribute; no weights changed.\n")
```
