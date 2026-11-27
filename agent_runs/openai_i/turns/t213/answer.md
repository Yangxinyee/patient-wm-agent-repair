The reduced head edit preserved factual fit but barely improved the dose responses, so it is not yet a satisfactory repair. I’ll revisit the action-to-state computation and the record-based effect estimates. A useful next pathway must change dose responses in the supported direction without exceeding the factual-loss limit.

```python
from pathlib import Path
import numpy as np, pandas as pd, torch, inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
notes189 = Path("/work/notes.md").read_text()
print("\n=== EARLY INVESTIGATION ===")
print(notes189[:18000])
print("\n=== RETAINED COMPUTATION HELPERS ===")
for name189, obj189 in list(globals().items()):
    if callable(obj189) and getattr(obj189, "__module__", "") == "__main__":
        if any(s in name189.lower() for s in ["replay", "step", "core", "readout", "encode", "feature", "register", "head"]):
            try:
                print(name189, inspect.signature(obj189),
                      "references:", obj189.__code__.co_names,
                      "locals:", obj189.__code__.co_varnames)
            except Exception:
                pass
print("\nActive edits:", edited())
with open("/work/notes.md", "a") as f189:
    f189.write("\n## Python 189 — revisit the controllable action pathway\n"
               "- No active edits; gain-.6 direct-head fallback is safe locally but insufficient.\n"
               "- Recovering architecture and original record evidence before another pathway test.\n")
```