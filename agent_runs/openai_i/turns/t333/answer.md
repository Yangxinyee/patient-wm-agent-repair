The empty-action correction cut the factual penalty to **0.00045 nats per reading**, well below the 0.005 limit. The edit is now a viable candidate, but its dose responses remain unvalidated. I’ll recover the record-derived targets and testing setup; matching those targets while preserving factual fit would support keeping it.

```python
import os, re, json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== DATA README ===")
print(open("/data/README.md").read())

notes287 = open("/work/notes.md").read()
print("=== EARLY NOTEBOOK ===")
print(notes287[:16000])
lines287 = notes287.splitlines()
hits287 = [
    i for i, s in enumerate(lines287)
    if re.search(r"target|true effect|causal estimate|instrument|identif|coefficient", s, re.I)
]
print("=== TARGET / IDENTIFICATION NOTES ===")
for i in hits287[-65:]:
    print("\n".join(lines287[max(0, i-1):min(len(lines287), i+3)]))

factual287 = pd.read_csv("/work/fwd_34/factual_original.csv")
print("=== FACTUAL COLUMNS ===", factual287.columns.tolist())
staycol287 = next(c for c in factual287.columns if "stay" in c.lower())
cohort287 = factual287[staycol287].drop_duplicates().tolist()
print("=== LATEST FACTUAL COHORT ===")
print(json.dumps([int(x) if isinstance(x, (int, np.integer)) else x for x in cohort287]))

print("=== SAVED CANDIDATES ===")
for stem287 in ["neutral209", "corrected269", "token257", "corrected_direct286"]:
    with np.load("/work/" + stem287 + ".npz", allow_pickle=False) as z287:
        print(stem287, [(k, z287[k].shape) for k in z287.files])

print("=== AVAILABLE HELPERS / VARIABLES ===")
print(sorted(k for k in list(globals())
             if not k.startswith("_") and
             (callable(globals()[k]) or
              re.search(r"target|effect|direction|response|neutral|route|dose|stay|replay|mean|beta", k, re.I))))
print("=== CURRENT EDITS ===", edited())

with open("/work/notes.md", "a") as f287:
    f287.write(
        "\n## Python 287 — recover direct-token validation context\n"
        "- corrected_direct286 factual dNLL is +0.00045 on the latest 100-stay cohort.\n"
        "- This clears the factual limit on that cohort, not yet causal validation.\n"
        "- Recovering architecture, record-derived targets, saved candidates, and stay IDs.\n"
        "- No weights changed.\n"
    )
```