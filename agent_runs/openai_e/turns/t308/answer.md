The reduced edit preserves a strong insulin response, but the norepinephrine responses remain asymmetric. I’ll recover the record-derived targets, factual-fit result, and prior accepted candidate. The reduced edit is worth keeping only if it improves the target agreement while staying within the NLL limit.

```python
from pathlib import Path
import numpy as np, pandas as pd, re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes264 = Path("/work/notes.md").read_text()
print("\n=== INITIAL RECORD EVIDENCE ===")
print(notes264[:6500])
print("\n=== RECENT ACCEPTED-CANDIDATE NOTES ===")
for run in (219, 236):
    matches = list(re.finditer(rf"(?m)^##[^\n]*\b{run}\b[^\n]*", notes264))
    for m in matches[-2:]:
        end = notes264.find("\n##", m.end())
        print(notes264[m.start(): end if end >= 0 else m.start()+2200][:2600])

print("\n=== FORWARD 31 FACTUAL FIT ===")
for model in ("original", "edited"):
    p = Path(f"/work/fwd_31/factual_{model}.csv")
    if p.exists():
        df264 = pd.read_csv(p)
        print(model, "shape", df264.shape, "columns", df264.columns.tolist())
        nll_cols264 = [c for c in df264.columns if "nll" in c.lower()]
        print(df264[nll_cols264].mean(numeric_only=True).to_dict())
    else:
        print(model, "files", [p.name for p in Path("/work/fwd_31").iterdir()])

print("\n=== SAVED CANDIDATE CONTRACTS ===")
for filename in ("joint236.npz", "nonlinear262.npz", "ne_nonlinear250.npz"):
    with np.load("/work/"+filename) as a:
        print(filename)
        for key in a.files:
            val = a[key]
            print(key, val.shape, val.dtype,
                  val.tolist() if val.size <= 8 else
                  {"norm": float(np.linalg.norm(val))} if np.issubdtype(val.dtype, np.number)
                  else "")
print("CURRENT EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 264: restart audit\n"
            "Forward32 reduced nonlinear rollout: insulin2=-11.289, insulinhalf=5.474, "
            "NE2=.103, NEhalf=-1.067 on tuning100. Recovering architecture, "
            "record-derived targets, forward31 factual fit, and accepted candidate contracts "
            "before retaining or rejecting the prelude edit. No weights changed.\n"
            "Python runs: 264; forwards: 32.\n")
```