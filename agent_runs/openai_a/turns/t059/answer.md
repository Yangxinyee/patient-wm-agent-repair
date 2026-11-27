The residual-dose decoders failed to generalize, so I would not use them for an edit yet. The observational slopes suggest missing dose sensitivity, but do not establish causality. Next I’ll recover the exact architecture and cached analysis to find a better-controlled edit. It must improve intervention responses while keeping factual NLL within the required tolerance.

```python
import os, json, numpy as np, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

print("\n=== RETAINED ARRAYS, FUNCTIONS, AND DICTIONARIES ===")
for name, value in sorted(list(globals().items())):
    if name.startswith("_"):
        continue
    if isinstance(value, (np.ndarray, torch.Tensor)):
        print(name, type(value).__name__, tuple(value.shape))
    elif callable(value) and getattr(value, "__module__", "") == "__main__":
        print(name, "function")
    elif isinstance(value, dict) and name not in ("SHAPES", "VOCAB"):
        keys = list(value.keys())
        print(name, "dict", len(keys), "keys:", str(keys[:12])[:220])

print("\n=== TENSOR SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])

notes = open("/work/notes.md").read()
print("\n=== NOTEBOOK SECTION TITLES ===")
print("\n".join(line for line in notes.splitlines() if line.startswith("#")))
print("\n=== EARLY IDENTIFICATION NOTES ===")
print(notes[:10000])
print("\nCURRENT EDITS:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart after Python 52; forward 4/50\n"
            "- Residual-dose feature decoders have negative held-out R2; not accepted as a safe repair.\n"
            "- Recovering exact architecture and prior identification evidence before editing.\n"
            "- Acceptance requires intervention improvement and factual NLL increase <= 0.005.\n")
```