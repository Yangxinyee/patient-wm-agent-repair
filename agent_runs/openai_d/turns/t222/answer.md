The decoder responds almost symmetrically to positive and negative changes, so it does not explain the weak norepinephrine half-dose response by itself. The stronger trial also failed factual-fit validation on a second cohort. I’ll recover the architecture, causal evidence, and probe code before changing weights; exact dose-bin probes should distinguish a representation problem from a gating problem.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

notes191 = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK / CAUSAL EVIDENCE ===")
print(notes191[:18000])

print("=== AVAILABLE FUNCTIONS ===")
for name191, obj191 in sorted(list(globals().items())):
    if inspect.isfunction(obj191) and not name191.startswith("_"):
        try:
            print(name191, inspect.signature(obj191))
        except Exception:
            print(name191)

print("=== RECENT / CAUSAL / PROBE VARIABLES ===")
prefixes191 = (
    "c183", "c185", "r165", "r170", "r171", "r176", "r178",
    "r182", "r183", "r187", "r188", "r189", "r190",
    "causal", "probe", "meta", "target"
)
for name191, obj191 in sorted(list(globals().items())):
    if name191.startswith(prefixes191):
        if isinstance(obj191, dict):
            print(name191, "dict", list(obj191.keys())[:35])
        elif hasattr(obj191, "shape"):
            print(name191, type(obj191).__name__, tuple(obj191.shape))
        elif isinstance(obj191, (list, tuple)):
            print(name191, type(obj191).__name__, "length", len(obj191),
                  "preview", repr(obj191[:8])[:500])
        elif not inspect.isfunction(obj191):
            print(name191, type(obj191).__name__, repr(obj191)[:400])
print("=== CURRENT EDITS ===", edited())
```