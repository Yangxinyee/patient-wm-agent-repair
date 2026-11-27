The perturbed-state export is complete, but it does not yet show whether the probe improves dose effects. I’ll restore the original embedding, recover the architecture and record-based estimates, and compare the two state exports. A useful probe should isolate dose sensitivity without substantially changing factual predictions.

```python
import os, pathlib, inspect, numpy as np, torch

probe_numeric = load("embed.numeric.weight")
numeric_original = load_original("embed.numeric.weight")
probe_delta = np.asarray(probe_numeric) - np.asarray(numeric_original)
print("Numeric probe:", np.asarray(probe_numeric).shape,
      "delta norm:", float(np.linalg.norm(probe_delta)))
print("Delta norms by input column:", np.linalg.norm(probe_delta, axis=0))
drop_edit("embed.numeric.weight")
print("Remaining edits:", edited())

print("\n=== ARCHITECTURE ===")
print(pathlib.Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY RECORD EVIDENCE ===")
notes_text = pathlib.Path("/work/notes.md").read_text()
print(notes_text[:11000])

print("\n=== FORWARD FILES ===")
for k in range(1, 9):
    p = pathlib.Path(f"/work/fwd_{k}")
    print(k, [(f.name, f.stat().st_size) for f in p.iterdir()] if p.exists() else [])

print("\n=== LIVE ARRAYS AND HELPERS ===")
inventory = []
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        inventory.append((name, type(obj).__name__, tuple(obj.shape)))
    elif inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            inventory.append((name, "function", str(inspect.signature(obj))))
        except Exception:
            inventory.append((name, "function", "?"))
    elif isinstance(obj, dict) and name not in {"__builtins__", "SHAPES", "VOCAB"}:
        inventory.append((name, "dict", len(obj), list(obj.keys())[:8]))
print(*inventory, sep="\n")

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 141\nForward 8 exported the temporary centered 1.5x numeric-dose probe states. Restored embed.numeric.weight to original; recovering architecture, causal evidence, and caches for an exact exported-state sensitivity comparison. Counts: 141 Python runs, 8 forward passes.\n")
```