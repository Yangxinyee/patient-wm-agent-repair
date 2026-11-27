The architecture separates drug inputs from a strong reading carryover mechanism. That could explain the weak rollout responses, but it does not yet identify which weights to change. I’ll recover the earlier gain tests and record-based estimates to avoid repeating failed experiments.

```python
from pathlib import Path
import numpy as np

notes = Path("/work/notes.md").read_text()
print("EARLY NOTEBOOK\n", notes[:6200])

print("\nAVAILABLE HELPERS")
print([n for n, x in list(globals().items())
       if callable(x) and getattr(x, "__module__", "") == "__main__"])

print("\nCURRENT OPERATIONS")
print(ops())

print("\nSAVED CANDIDATES")
if "repair_candidates" in globals():
    for name, candidate in repair_candidates.items():
        print(name, type(candidate).__name__,
              list(candidate.keys()) if isinstance(candidate, dict) else f"length={len(candidate)}")

print("\nSELECTED PERSISTENT VALUES")
for name in ["baseline_nll", "baseline_rollouts", "dynamic_moments",
             "tuning_ids", "probe_ids", "train_ids", "dose_directions",
             "PYTHON_RUNS", "FORWARD_PASSES"]:
    if name not in globals():
        continue
    obj = globals()[name]
    if isinstance(obj, dict):
        print(name, {k: ({"shape": tuple(v.shape), "norm": float(np.linalg.norm(v))}
                        if hasattr(v, "shape") else v)
                     for k, v in obj.items()})
    elif hasattr(obj, "shape"):
        print(name, "shape", obj.shape, "first", np.asarray(obj).reshape(-1)[:12].tolist())
    elif isinstance(obj, (list, tuple)):
        print(name, "length", len(obj), "first", obj[:12])
    else:
        print(name, obj)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Restart / Python runs 49–50\n"
            "- Re-read the architecture. Only numeric gain 4 is installed.\n"
            "- Direct reading corrections tested so far have weak rollout benefit or violate factual fit.\n"
            "- Recovering previous gain tests and causal evidence before selecting a new pathway.\n"
            "- Forward passes used: 22.\n")
```
