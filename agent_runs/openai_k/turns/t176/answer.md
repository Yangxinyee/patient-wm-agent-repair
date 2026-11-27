The last assay showed that the proposed correction needs unusually large token changes and then saturates, so its linear calibration is unreliable. I will leave it unapplied and inspect the exact prediction path and retained effect estimates. The next approach must reproduce official factual predictions and produce dose responses without large state perturbations; failure on either test would refute it.

```python
from pathlib import Path
import inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RELEVANT RETAINED FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and not name.startswith("_"):
        if any(k in name.lower() for k in
               ("head", "predict", "query", "update", "candidate", "contrast",
                "dose", "innovation", "register", "effect", "feature")):
            try:
                sig = str(inspect.signature(obj))
            except Exception:
                sig = "(signature unavailable)"
            print(name, sig)

print("\n=== RETAINED INTERFACES ===")
for name in ("W144", "M143", "R143", "F143o", "F143e", "S143o",
             "readoutweights151", "candidate158", "candidate139"):
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, "keys:", list(obj.keys())[:35])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        else:
            print(name, type(obj).__name__)

print("\n=== EARLY RECORD-IDENTIFICATION NOTES ===")
notes160 = Path("/work/notes.md").read_text()
sections160 = notes160.split("\n## ")
matches160 = [
    s for s in sections160
    if any(k in s.lower() for k in
           ("cross-fit", "crossfit", "identified", "persistence", "structural",
            "instrument", "true effect", "causal effect", "target rollout"))
]
for s in matches160[:12]:
    print("\n## " + s[:2200])

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 160: exact-head redesign\n"
            "- Rejected candidate158 because token magnitude and nonlinear saturation invalidate its linear calibration.\n"
            "- Recovering the actual prediction path and record-based dynamic effects before further editing.\n"
            "- Saved edits remain candidate139; Python runs: 160; forward passes: 10.\n")
```
