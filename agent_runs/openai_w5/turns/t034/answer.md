The architecture confirms that doses affect readings through the recurrent state, while the value head also relies on the last-reading register. I’ll recover the existing reconstruction helpers and causal-analysis objects without another long tensor listing. Those should help test a targeted edit that changes dose responses but leaves ordinary trajectories nearly unchanged.

```python
import inspect, json

print("--- Remaining state/head tensor shapes ---")
for name in NAMES:
    if name.startswith(("z_", "last_", "query_", "gap", "type_", "filter_", "pending_")):
        print(name, SHAPES[name])

print("\n--- Functions ---")
for key in sorted(list(globals())):
    obj = globals()[key]
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(key, sig)

print("\n--- Dataframes and arrays ---")
for key in sorted(list(globals())):
    obj = globals()[key]
    if hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(key, obj.shape, ",".join(map(str, list(obj.columns)[:28])))
    elif isinstance(obj, np.ndarray) and not key.startswith("_"):
        print(key, obj.shape)

print("\n--- Dictionaries ---")
for key in sorted(list(globals())):
    obj = globals()[key]
    if isinstance(obj, dict) and key not in {"__builtins__", "SHAPES", "VOCAB"}:
        print(key, list(obj.keys())[:18])

print("\n--- Vocabulary structure ---")
print(list(VOCAB))
for key in ("codes", "config", "normalization", "normalisation", "categoricals"):
    if key in VOCAB:
        text = json.dumps(VOCAB[key])
        print(key, text[:1500])

print("\n--- Saved forward files ---")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
```
