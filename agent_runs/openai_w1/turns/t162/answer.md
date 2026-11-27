The dose probe barely changed predictions, and amplifying value-head sensitivity appears likely to disturb factual fit. The record-based effect estimates remain provisional. I’ll inspect the architecture and cached computations to find a pathway we can edit and compensate; it must produce dose responses without exceeding the factual-NLL budget.

```python
import os, inspect, json, numpy as np, torch
print("CURRENT EDITS:", edited())
for path in ["/model/ARCHITECTURE.md", "/data/README.md"]:
    text = open(path).read()
    print("\nFILE:", path, "CHARS:", len(text))
    print(text[:16000])
print("\nLIVE HELPERS:")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(?)"
        print(name, sig)
print("\nLIVE ARRAYS / DICTS:")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"SHAPES", "VOCAB", "text"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape), str(obj.dtype))
    elif isinstance(obj, dict):
        keys = list(obj)
        print(name, "dict", len(keys), "keys:", str(keys[:8])[:230])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 148\nRestart recovery. Probe/factual covariance bounds argue against blindly amplifying the value head. Reading architecture and bounded live-cache inventory before constructing a compensated, reversible dose-path edit. No edits. Counts: 148 Python runs, 8 forward passes.\n")
```