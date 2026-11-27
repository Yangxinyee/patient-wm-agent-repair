The kernel still contains the posterior and recurrent reconstruction helpers. I’ll inspect their inputs and the saved state samples next, then measure how a small latent perturbation changes glucose and MAP predictions. Drug-specific effects would support this route; mixed or negligible effects would argue against it.

```python
print("--- RECENT GLOBAL NAMES ---")
print(list(globals())[-150:])
print("\n--- RECONSTRUCTION OBJECTS ---")
for k in ["sm_repair", "sx_repair", "Wrepair", "core_rows104", "all_e104",
          "all_u104", "ctx104", "z104", "h104", "memory104",
          "head_cfg", "best_cfg", "cfg", "S", "head_data", "benchmark_stays"]:
    if k not in globals():
        continue
    obj = globals()[k]
    print(k, type(obj).__name__, getattr(obj, "shape", ""))
    if isinstance(obj, pd.DataFrame):
        print(obj.head(2).to_string(index=False))
    elif isinstance(obj, dict):
        print(list(obj.items())[:3] if k != "Wrepair" else list(obj)[:8])
print("\n--- RECENT ARRAY SHAPES ---")
for k, v in list(globals().items()):
    if any(s in k for s in ["repair", "100", "101", "102", "103", "104"]):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, tuple(v.shape))
        elif isinstance(v, (list, tuple)):
            print(k, "length", len(v))
print("\n--- RECONSTRUCTION FUNCTION DETAILS ---")
for name in ["step104", "memory104", "previous_context_repair"]:
    f = globals()[name]
    print(name, "constants:", f.__code__.co_consts)
    print(name, "references:", f.__code__.co_names)
    try:
        print(inspect.getsource(f))
    except Exception:
        print("Source text unavailable.")
```
