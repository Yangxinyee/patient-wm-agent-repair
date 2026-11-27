The last result shows that glucose and MAP observations map to distinct, reproducible latent directions. That identifies a possible editing pathway, but does not establish drug effects. I’ll recover the architecture and existing causal estimates before constructing a reversible edit; improved dose responses with unchanged factual NLL would support that approach.

```python
from pathlib import Path
import inspect, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLY EVIDENCE ===")
print(notes_text[:11000])

print("\n=== RETAINED ARRAYS / TABLES ===")
for key, value in sorted(list(globals().items())):
    if key.startswith("_"):
        continue
    if hasattr(value, "shape") and not callable(value):
        print(key, type(value).__name__, tuple(value.shape))
    elif isinstance(value, dict) and any(
        term in key.lower() for term in
        ["dose", "caus", "latent", "post", "direction", "ridge", "coef", "panel", "state"]
    ):
        print(key, "dict", list(value.keys())[:12])

print("\n=== RETAINED HELPERS ===")
for key, value in sorted(list(globals().items())):
    if inspect.isfunction(value) and value.__module__ == "__main__":
        try:
            print(key, inspect.signature(value))
        except Exception:
            print(key)

print("\n=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python 105 — restart and edit-path recovery\n"
            "- Observation-linked latent directions are reproducible, but are not causal estimates.\n"
            "- Recovering architecture, retained helpers, and early record evidence before a reversible edit.\n"
            "- No weights changed.\n")
```