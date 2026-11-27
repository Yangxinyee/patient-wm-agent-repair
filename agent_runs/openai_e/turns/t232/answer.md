The architecture confirms that the rejected edit changed recurrent state updates, not just MAP predictions. Its training-state safeguards did not preserve factual fit. I’ll restore the saved insulin baseline and recover the measured trade-offs; another NE edit is worthwhile only if it improves on those validated candidates.

```python
import torch
from pathlib import Path
import numpy as np

def as_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

base_delta200 = np.load("/work/baseline77.npz")["delta"].copy()
orig_b200 = as_np(load_original("core.b_in.weight")).copy()
save_edit("core.b_in.weight", torch.from_numpy(orig_b200 + base_delta200))
print("Restored baseline77; maximum reconstruction error:",
      np.max(np.abs(as_np(load("core.b_in.weight")) - orig_b200 - base_delta200)))

sections200 = Path("/work/notes.md").read_text().split("\n## ")
print("\n=== EXPERIMENT HEADINGS ===")
for s in sections200:
    h = s.splitlines()[0]
    if any(str(k) in h for k in [63, 77, 78, 82, 103, 107, 148, 151, 154, 155, 159, 170, 181]):
        print(h)

print("\n=== SELECTED VALIDATION NOTES ===")
for s in sections200:
    h = s.splitlines()[0]
    if any(t in h for t in ["Run 77:", "Run 107:", "Run 151:", "Run 154:", "Run 155:",
                            "Runs 76", "Runs 106", "Runs 150", "Runs 154"]):
        print("\n## " + s[:1400])

print("\n=== FORWARD CSV SCHEMAS ===")
for k in [1, 7, 9, 10, 14, 18, 20, 21, 23]:
    p = Path(f"/work/fwd_{k}")
    for f in sorted(p.glob("*.csv")):
        import pandas as pd
        d = pd.read_csv(f)
        print(k, f.name, len(d), list(d.columns))
        if "dose_edit" in f.name:
            print(d.head(2).to_string(index=False))

print("\n=== RELEVANT PERSISTENT OBJECTS ===")
for name in sorted(list(globals())):
    if any(t in name.lower() for t in ["causal", "trial_direction", "feature", "contrast", "ne_fit", "target", "ids"]):
        obj = globals()[name]
        if isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:8])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, len(obj))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 201: restore insulin baseline\n"
            "Restored original core.b_in.weight plus baseline77 delta. "
            "Candidate199 remains archived but rejected for factual cost=.02270. "
            "Recovering prior validation trade-offs and persistent fitting objects. "
            "Python runs: 201; forwards: 23.\n")
```
