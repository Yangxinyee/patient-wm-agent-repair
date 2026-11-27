This is still the truncated inventory; no new validation or edit has happened. I’ll narrow the inspection further to the head helper and its caches. If they expose aligned rows and hidden features, the next step will be a direct original-versus-probe comparison.

```python
import inspect, pathlib, pandas as pd

print("HEAD HELPER:")
print("signature:", inspect.signature(ce_make_heads))
print("references:", ce_make_heads.__code__.co_names)
print("locals:", ce_make_heads.__code__.co_varnames[:35])

print("\nSELECTED CACHES:")
for name in ["he_s0", "he_base", "he_kernels", "ce_head", "ce_meta", "panel"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, "keys:", list(obj.keys())[:18])
        for key, value in list(obj.items())[:5]:
            if hasattr(value, "shape"):
                print(" ", key, tuple(value.shape))
            elif isinstance(value, dict):
                print(" ", key, "dict", list(value.keys())[:5])
            else:
                print(" ", key, repr(value)[:100])

print("\nHEAD ARRAYS:")
for name in ["he_train_ids", "he_phi0", "he_p0", "he_centres", "he_mean0",
             "he_bins", "he_nll0", "he_yr", "he_dr", "he_kk"]:
    obj = globals().get(name)
    if hasattr(obj, "shape"):
        print(name, tuple(obj.shape), str(obj.dtype))

print("\nDATAFRAMES:")
for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, list(obj.columns)[:14])

print("\nFORWARD FILES:")
for k in range(1, 9):
    p = pathlib.Path(f"/work/fwd_{k}")
    print(k, [f.name for f in p.iterdir()] if p.exists() else [])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 143\nFocused head-helper and cache inspection; no weights edited. Counts: 143 Python runs, 8 forward passes.\n")
```