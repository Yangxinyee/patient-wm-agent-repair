The notes support next-hour insulin effects near −2 per rate unit and norepinephrine effects near +3, while insulin’s long-horizon persistence remains uncertain. They also document unresolved head-reconstruction error. I’ll check the cached implementations and whether prior-sampling variability could have destabilized the rejected correction.

```python
import inspect
import numpy as np

print("VOCAB KEYS:", list(VOCAB))
print("CONFIG:", VOCAB.get("config"))
print("CODES:", VOCAB.get("codes"))

print("\nRELEVANT TENSORS")
for name in NAMES:
    if any(term in name for term in
           ("attr_heads.value_bin", "z_prior", "z_head", "gap",
            "out_norm", "query_norm", "last_numeric",
            "last_status", "type_emb")):
        print(name, SHAPES[name])

print("\nFUNCTION CONVENTIONS")
for name in ("reconstruct_reading_head", "score_view_local",
             "activation", "ln", "mlp"):
    obj = globals()[name]
    params = []
    for p in inspect.signature(obj).parameters.values():
        d = p.default
        if d is inspect.Parameter.empty:
            desc = "required"
        elif isinstance(d, (str, int, float, bool, type(None))):
            desc = repr(d)
        else:
            desc = type(d).__name__ + str(getattr(d, "shape", ""))
        params.append((p.name, desc))
    print(name, params)
    print("constants:", [c for c in obj.__code__.co_consts
                          if isinstance(c, (str, int, float, tuple))])

print("\nCORE RECONSTRUCTION CHECKS")
print(core_check_table.sort_values(core_check_table.columns[-1]).head(8).to_string(index=False))

print("\nREJECTED DECODER MAGNITUDE")
for name in ("coef", "coef_d", "b0", "head_hidden", "baseline_probs_mc",
             "fresh_eps", "head_probs"):
    obj = globals().get(name)
    if obj is not None:
        arr = np.asarray(obj)
        print(name, arr.shape, "RMS", np.sqrt(np.nanmean(arr**2)),
              "maxabs", np.nanmax(np.abs(arr)))

print("\nPOSTERIOR CONTRAST CV")
print(pd.read_csv("/work/posterior_contrast_cv.csv").head(10).to_string(index=False))
```
