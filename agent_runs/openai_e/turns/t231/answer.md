The new norepinephrine edit worsened factual NLL by **0.02270 nats**, well beyond the 0.005 limit, so it is not acceptable. I’ll recover the saved baseline and prior evidence before choosing another edit. The next candidate must improve dose responses while preserving factual fit on independent stays.

```python
from pathlib import Path
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== SAVED BASELINE AND CANDIDATE SCHEMAS ===")
for fn in ["baseline77.npz", "ne_candidate199.npz", "ne_reduced107.npz"]:
    with np.load("/work/" + fn) as a:
        print(fn, {k: (a[k].shape, str(a[k].dtype)) for k in a.files})

notes = Path("/work/notes.md").read_text()
sections = notes.split("\n## ")
terms = ["causal", "instrument", "random", "baseline77", "Run 77",
         "Run 107", "Run 151", "Run 154", "Run 155", "true", "persistence"]
hits = [s for s in sections if any(t.lower() in s.lower() for t in terms)]
print("\n=== PRIOR EVIDENCE (BOUNDED) ===")
print("\n## ".join(hits)[-22000:])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 200: reject candidate199 factual fit\n"
            "Forward23: original NLL=1.63461, candidate199=1.65731; "
            "cost=.02270 exceeds .005. Candidate199 rejected. "
            "Recovering architecture, baseline schema, and prior causal/validation evidence "
            "before restoring validated weights. Python runs: 200; forwards: 23.\n")
```