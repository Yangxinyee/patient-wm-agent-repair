The architecture confirms that the value head can be edited without directly changing the other heads on factual histories. The inventory was cut off before reaching the new feature tables, so I’ll inspect those interfaces narrowly. This will tell us whether a constrained correction is feasible before touching any weights.

```python
import inspect, pandas as pd

snapshot = list(globals().items())
print("=== NEWEST VARIABLES ===")
for name, obj in snapshot[-85:]:
    if name.startswith("_"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape), str(obj.dtype))
    elif isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, list(obj.columns))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), repr(list(obj.keys())[:7])[:180])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj),
              repr(obj[:2])[:160] if len(obj) < 20 else "")
    elif inspect.isfunction(obj):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name, "function")
    elif isinstance(obj, (int, float, str)):
        print(name, repr(obj)[:140])

print("\n=== LOCAL RECONSTRUCTION HELPERS ===")
funcs = [(n, o) for n, o in snapshot
         if inspect.isfunction(o) and getattr(o, "__module__", "") == "__main__"]
for name, obj in funcs[-28:]:
    print(name, inspect.signature(obj))

print("\n=== VALUE-HEAD TENSORS ===")
for name in NAMES:
    if "attr_heads.value_bin" in name:
        w = load_original(name)
        print(name, tuple(w.shape), "RMS", float(torch.as_tensor(w).square().mean().sqrt()))

print("\n=== IDENTIFIED EFFECTS ===")
for name in ("dml_results", "dynamic_ir", "factual_missing_effect0"):
    if name in globals():
        print(name, repr(globals()[name])[:1500])

print("\n=== LATEST NOTEBOOK ENTRIES ===")
print(open("/work/notes.md").read()[-1800:])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 61: focused dataset inspection\n"
            "- Architecture confirms value-head-only edits leave other factual heads directly unchanged.\n"
            "- Previous broad inventory was truncated; inspecting newest tables and reconstruction helpers only.\n"
            "- Python runs: 61; forward passes: 4; no tensors edited.\n")
```