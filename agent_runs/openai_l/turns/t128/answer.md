The last test found quiet state directions with reliable readouts, which could carry a correction without disturbing factual predictions. I’ll recover the architecture and existing helpers before wiring that route. It succeeds only if it responds to dose changes while remaining quiet on factual records.

```python
import os, json, inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== DATA README ===")
print(open("/data/README.md").read())
print("\n=== NOTEBOOK: EARLY FINDINGS ===")
notes119 = open("/work/notes.md").read()
print(notes119[:12000])

print("\n=== AVAILABLE FUNCTIONS ===")
for name119, obj119 in list(globals().items()):
    if callable(obj119) and getattr(obj119, "__module__", None) == "__main__":
        try:
            print(name119, inspect.signature(obj119))
        except Exception:
            print(name119)

print("\n=== ARRAY/TENSOR VARIABLES ===")
for name119, obj119 in list(globals().items()):
    if isinstance(obj119, (np.ndarray, torch.Tensor)):
        print(name119, tuple(obj119.shape), str(obj119.dtype))

print("\n=== OTHER STATE NAMES ===")
print([k for k, v in list(globals().items())
       if not k.startswith("_") and not callable(v)
       and not isinstance(v, (np.ndarray, torch.Tensor))
       and k != "notes119"])
print("\nCurrent edits:", edited())

with open("/work/notes.md", "a") as f119:
    f119.write("\n## Run 119 / restart recovery\n"
               "Private state directions have factual RMS≈0.013 and normalized "
               "readout SD≈0.03–0.04, with near-unit response across all four "
               "measurement types. Recovering architecture, causal evidence, "
               "and helpers to implement and validate a selective dose-effect "
               "state channel. No weights edited.\n")
```