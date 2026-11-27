The architecture allows dose-sensitive head edits and separate factual corrections, but the diagnostic output was truncated before the useful details appeared. I’ll retrieve smaller summaries of the causal estimates and validation results. These should tell us whether to reduce the current edit or restore a stronger validated fallback.

```python
from pathlib import Path
import numpy as np, pandas as pd, inspect, re

n260_notes = Path("/work/notes.md").read_text()
print("=== EARLY RECORD-BASED FINDINGS (bounded) ===")
print(n260_notes[:5800])

print("\n=== RECENT PAIRED FACTUAL DELTAS ===")
n260_recent = []
for k in range(29, 39):
    p = Path(f"/work/fwd_{k}")
    if (p/"factual_original.csv").exists() and (p/"factual_edited.csv").exists():
        a, b = pd.read_csv(p/"factual_original.csv"), pd.read_csv(p/"factual_edited.csv")
        cols = [c for c in a if "nll" in c.lower() and c in b]
        s = next((c for c in ["stay_id", "stay"] if c in a), None)
        for c in cols:
            rec = (k, len(a), None if s is None else (a[s].min(), a[s].max()),
                   c, float(b[c].mean()-a[c].mean()))
            n260_recent.append(rec)
            print(rec)

print("\n=== AVAILABLE FUNCTION NAMES ===")
print(", ".join(k for k,v in list(globals().items())
                if inspect.isfunction(v) and not k.startswith("_"))[:1800])

print("\n=== RECENT CANDIDATE VARIABLE NAMES ===")
print(", ".join(k for k in list(globals())
                if re.match(r"r(23[5-9]|24[0-9]|25[0-8])", k))[:1100])

print("\n=== SAVED CANDIDATE KEYS ===")
for filename in ["safe_6288c226d6.npz", "cancellation50_f33.npz", "lastbin80_scale4.npz"]:
    with np.load("/work/"+filename) as z:
        print(filename, [(k, z[k].shape) for k in z.files][:12])
```
