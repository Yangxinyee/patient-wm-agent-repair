The nonlinear gate edits badly damaged factual fit, so I reject them; they were not saved. The current passing edit remains intact. Next I’ll recover the record-derived targets and repair interfaces to choose a more controlled adjustment. A useful candidate must improve the dose readouts while keeping the NLL increase below 0.005.

```python
from pathlib import Path
import inspect, re, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes226 = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND TARGETS ===")
print(notes226[:15000])

print("\n=== RELEVANT KERNEL OBJECTS ===")
keys226 = sorted(k for k in list(globals())
                 if not k.startswith("_") and
                 re.search(r"target|kernel|causal|effect|replay|candidate|backup|ridge|channel|impulse|dual|r225|passing|weights|head_eval", k, re.I))
for k226 in keys226:
    v226 = globals()[k226]
    if callable(v226):
        try:
            desc226 = str(inspect.signature(v226))
        except Exception:
            desc226 = "callable"
    elif isinstance(v226, dict):
        desc226 = "dict keys=" + repr(list(v226.keys())[:16])
    elif hasattr(v226, "shape"):
        desc226 = f"{type(v226).__name__} shape={tuple(v226.shape)}"
    else:
        desc226 = repr(v226)[:180]
    print(k226, desc226)

with open("/work/notes.md", "a") as f226:
    f226.write("\n\n## Run 226\nRejected unsaved nonlinear paired-gate candidates from Run 225: factual losses increased by 1.98–3.53 nats. Recovering architecture, record-based targets and retained repair interfaces for a controlled, factual-safe adjustment. No saved weights changed.\n")
```