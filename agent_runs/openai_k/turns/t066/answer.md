The reconstructed model closely matches the earlier probes, and we now have factual and dose-edited features without changing any weights. Next I’ll inspect the head layout and saved tables so I can fit a correction that preserves factual predictions. The key test will be whether dose responses improve without exceeding the factual-NLL limit.

```python
import inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== RETAINED DATA AND HELPERS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape), str(obj.dtype))
    elif isinstance(obj, dict):
        keys = list(obj.keys())
        print(name, "dict", len(keys), "keys:", repr(keys[:12])[:240])
    elif isinstance(obj, (list, tuple)) and len(obj):
        first = obj[0]
        if isinstance(first, (dict, np.ndarray, torch.Tensor, tuple)):
            print(name, type(obj).__name__, len(obj),
                  "first:", repr(first)[:180] if isinstance(first, dict)
                  else (tuple(first.shape) if hasattr(first, "shape") else repr(first)[:180]))
    elif inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, "function", str(inspect.signature(obj)))
        except Exception:
            print(name, "function")

print("\n=== EARLY IDENTIFICATION NOTES ===")
notes = open("/work/notes.md").read()
print(notes[:9000])
print("\nCurrently edited:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 60: recover preservation-fit interfaces\n"
            "- Expanded core matches validated probes; 4,035 factual feature rows and 4,000 counterfactual rows retained.\n"
            "- Inspecting exact head layout and dataset interfaces before a preservation-constrained, non-gradient fit.\n"
            "- Python runs: 60; forward passes: 4; no tensors edited.\n")
```