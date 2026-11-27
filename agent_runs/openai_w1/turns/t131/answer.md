The last test resolved the reconstruction error: an empty action uses the encoder’s normalized empty vector, not zero. Reading-bundle posteriors now match the exported states to numerical precision. I’ll recover the architecture and quantitative effect estimates before constructing a fresh edit; disagreement with either would rule out reusing old candidates.

```python
from pathlib import Path
import inspect, numpy as np, torch

rx_arch = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(rx_arch)

rx_notes = Path("/work/notes.md").read_text()
print("\n=== QUANTITATIVE CAUSAL EVIDENCE ===")
for s in rx_notes.split("\n## "):
    if any(t in s.lower() for t in [
        "deconvol", "one-hour beta", "withheld-assay", "sequential-exogeneity",
        "original weak rollouts", "six-hour"
    ]) and len(s) < 4500:
        print("## " + s[:3500])

print("\n=== FRESH RECONSTRUCTION HELPERS ===")
for k in sorted(list(globals())):
    if k.startswith("rr_") and callable(globals()[k]):
        v = globals()[k]
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        print(k, sig)
        if hasattr(v, "__code__"):
            print("  globals:", v.__code__.co_names)

print("\n=== RELEVANT WEIGHTS ===")
for n in NAMES:
    if any(s in n.lower() for s in ["posterior", "prior", "value", "core", "action"]):
        print(n, SHAPES[n])

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 120\nRun 119 resolved posterior reconstruction: empty actions must use the normalized empty-set encoding. Reading and drug posteriors now agree with exported states to numerical precision. Recovering architecture and quantitative effect estimates before constructing a fresh, reversible edit. No weights edited. Counts: 120 Python runs, 6 forward passes.\n")
```