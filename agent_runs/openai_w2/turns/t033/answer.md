The architecture shows that doses enter through the recurrent state, while prediction heads also use the last recorded reading. Simply amplifying dose sensitivity could therefore damage factual fit. I’ll recover the saved features and effect estimates, then check probe alignment and state changes before choosing a correction.

```python
import json, inspect
from pathlib import Path

inventory_parts = []
for k, v in sorted(list(globals().items())):
    if k.startswith("_"):
        continue
    if inspect.isfunction(v):
        try:
            inventory_parts.append(f"{k}{inspect.signature(v)}")
        except Exception:
            inventory_parts.append(f"{k}(function)")
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        inventory_parts.append(f"{k}:{type(v).__name__}{tuple(v.shape)}")
    elif hasattr(v, "columns") and hasattr(v, "shape"):
        inventory_parts.append(f"{k}:{type(v).__name__}{v.shape}:{list(v.columns)}")
    elif isinstance(v, dict) and k not in ("__builtins__",):
        inventory_parts.append(f"{k}:dict({len(v)}):{list(v.keys())[:12]}")
print("=== INVENTORY ===")
print("\n".join(inventory_parts)[:4500])

print("\n=== FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\n=== VOCAB KEYS ===")
print(list(VOCAB.keys()))
print("Config:", VOCAB.get("config"))
print("\n=== HEAD WEIGHT NAMES ===")
print([(n, SHAPES[n]) for n in NAMES
       if any(s in n for s in ("attr_heads.value", "last_numeric", "query_norm", "z_head"))])

print("\n=== REFERENCE TYPE / PROBE ALIGNMENT ===")
if "S_reference" in globals():
    print(type(S_reference))
    if hasattr(S_reference, "keys"):
        print({k: getattr(S_reference[k], "shape", None) for k in S_reference.keys()})
        if all(k in S_reference for k in ("x", "stay_id", "hour")):
            probe_aligned = (
                np.array_equal(S_reference["stay_id"], S_probe["stay_id"])
                and np.allclose(S_reference["hour"], S_probe["hour"])
            )
            print("Aligned:", probe_aligned)
            if probe_aligned:
                state_probe_delta = S_probe["x"] - S_reference["x"]
                print("RMS differences, out/h/z:",
                      [float(np.sqrt(np.mean(state_probe_delta[:, a:b]**2)))
                       for a, b in [(0,192),(192,960),(960,992)]])

notes_recovered = Path("/work/notes.md").read_text()
print("\n=== EARLIER ANALYSIS (characters 4000–7800) ===")
print(notes_recovered[4000:7800])
```