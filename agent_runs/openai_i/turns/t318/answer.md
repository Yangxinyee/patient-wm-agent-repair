The fallback is restored. The recovered analyses show strong confounding: raw dose associations reverse after adjusting for pre-dose history, with estimated immediate effects near −2 for insulin and +3 for norepinephrine. I’ll recover the longer-term targets and test the corrected route at 80% response strength. This succeeds only if factual NLL stays within 0.005 while dose responses remain substantially improved.

```python
import pandas as pd, re
from pathlib import Path
import numpy as np

print("FUNCTIONS:", sorted(k for k,v in list(globals().items())
      if callable(v) and getattr(v,"__module__","") == "__main__"))
print("VARIABLES:", sorted(k for k in list(globals())
      if any(s in k.lower() for s in ["target","causal","compens","route","direction","275","276"])))

lines278 = Path("/work/notes.md").read_text().splitlines()
hits278 = [i for i,s in enumerate(lines278)
           if any(t in s.lower() for t in
                  ["target shift", "true effect", "rollout target", "persistence",
                   "distributed lag", "cumulative", "neutral209", "python 269"])]
for i in hits278[:22]:
    print("NOTE:", " | ".join(lines278[max(0,i-1):min(len(lines278),i+4)])[:550])

b278 = {k: np.array(v) for k,v in np.load("/work/neutral209.npz").items()}
c278 = {k: np.array(v) for k,v in np.load("/work/corrected269.npz").items()}
for k in c278:
    ref = b278[k] if k in b278 else load_original(k)
    d = c278[k] - ref
    print("DIFF", k, "norm", round(float(np.linalg.norm(d)),5),
          "max", round(float(np.max(np.abs(d))),5))
print("Z_IN_COLUMN_DIFFS:", [(j,round(float(np.linalg.norm(
      c278["z_in.weight"][:,j]-b278["z_in.weight"][:,j])),5))
      for j in range(32) if np.linalg.norm(
      c278["z_in.weight"][:,j]-b278["z_in.weight"][:,j]) > 1e-6])

cand278 = {k:v.copy() for k,v in c278.items()}
cand278["z_in.weight"] = b278["z_in.weight"] + 0.8 * (
    c278["z_in.weight"] - b278["z_in.weight"])
for k in list(edited()):
    drop_edit(k)
for k,v in cand278.items():
    save_edit(k,v)
np.savez("/work/gain278.npz", **cand278)

f278 = pd.read_csv("/work/fwd_28/factual_original.csv")
print("FACTUAL_COLUMNS", f278.columns.tolist())
sid278 = next(k for k in ["stay_id","stay","id"] if k in f278.columns)
dev_ids278 = f278[sid278].drop_duplicates().tolist()
print("VALIDATION_STAYS", dev_ids278)
with open("/work/notes.md","a") as f:
    f.write("\n## Python 278 — corrected route at response gain 0.8\n"
            "- Restored corrected269 gate and routing, scaled its z_in response delta to 0.8 relative to neutral209.\n"
            "- Saved gain278; provisional pending GPU factual and dose validation.\n")
```
