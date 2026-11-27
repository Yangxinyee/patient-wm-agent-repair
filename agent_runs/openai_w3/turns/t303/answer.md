The new edit increased factual value NLL by only **0.00096 nats**, below the 0.005 limit. That passes the first safeguard, but does not establish that its dose rollouts are correct. I’ll recover the saved rollout stays and record-based effect estimates, then test the actual rollout against those estimates.

```python
from pathlib import Path
import re, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD DOCUMENTATION ===")
print(Path("/data/README.md").read_text())

notes269 = Path("/work/notes.md").read_text()
print("=== EARLY IDENTIFICATION NOTES ===")
print(notes269[:6500])
hits269 = [
    line for line in notes269.splitlines()
    if re.search(r"reference|identified|instrument|causal|truth|target effects|effect estimates", line, re.I)
]
print("=== SAVED EVIDENCE / REFERENCES ===")
for line in hits269[:14] + hits269[-8:]:
    print(line[:1400])

print("=== STAY-ID VARIABLES ===")
for name, obj in list(globals().items()):
    if re.search(r"stay|ids", name, re.I) and not name.startswith("_"):
        if isinstance(obj, (list, tuple, np.ndarray)):
            a = np.asarray(obj)
            print(name, "shape", a.shape, "dtype", a.dtype,
                  "values", a.tolist() if a.ndim == 1 and a.size <= 100 else "omitted")
print("=== CURRENT CANDIDATE / ROLLBACK VARIABLES ===")
for name, obj in list(globals().items()):
    if re.search(r"267|checkpoint222|checkpoint249|checkpoint268|candidate268", name):
        if isinstance(obj, dict):
            print(name, {str(k): (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                         for k, v in obj.items()})
        elif hasattr(obj, "shape"):
            print(name, tuple(obj.shape))
        else:
            print(name, str(obj)[:700])

with Path("/work/notes.md").open("a") as f:
    f.write("\nPython run 269; forwards 21. Total-response gamma=.1 candidate passes initial factual test: original 1.63982, edited 1.64078, delta +.00096 nats on 100 tuning stays. Recovering exact rollout stay IDs and record-derived references before paired dose-rollout validation. No weights changed.\n")
```