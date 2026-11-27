The posterior matches most measurement bundles, but not those containing `glucose_lab`; the recurrent update also still has mismatches. These are reconstruction errors, not evidence about drug effects. I’ll recover the encoding details and existing helpers before fitting synthetic dose contrasts. Matching the saved states is the necessary check.

```python
from pathlib import Path
import types
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
for key, val in sorted(list(globals().items())):
    if isinstance(val, types.FunctionType) and not key.startswith("_"):
        print(key, val.__code__.co_varnames[:val.__code__.co_argcount])

print("\n=== RETAINED ARRAYS / TABLES / SMALL CONFIGURATIONS ===")
for key, val in sorted(list(globals().items())):
    if key.startswith("_"):
        continue
    if isinstance(val, np.ndarray):
        print(key, "array", val.shape)
    elif hasattr(val, "columns") and hasattr(val, "shape"):
        print(key, "table", val.shape, list(val.columns)[:18])
    elif isinstance(val, dict) and any(s in key.lower() for s in
                                      ("spec", "config", "graft", "check", "causal", "rd")):
        print(key, repr(val)[:1800])

print("\n=== EARLY IDENTIFICATION NOTES ===")
print(Path("/work/notes.md").read_text()[:16000])

with open("/work/notes.md", "a") as f:
    f.write("\n## Reconstruction audit; Python 126, forward 11/50\n"
            "- Most measurement posterior encodings match; glucose_lab bundles remain mismatched.\n"
            "- Recurrent transition reconstruction is not yet validated.\n"
            "- Recovering architecture, helpers, and earlier causal evidence; no weight changes.\n")
```
