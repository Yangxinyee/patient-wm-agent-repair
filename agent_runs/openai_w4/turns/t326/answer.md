The asymmetric insulin edit is saved, with a CPU factual penalty of +0.00395—within the limit—but its rollout effects remain unvalidated. I’ll recover the record-based effect estimates and validation setup before testing it. Agreement with those estimates and a GPU factual penalty below 0.005 would support keeping the edit; either failure would argue against it.

```python
from pathlib import Path
import re, inspect
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes281 = Path("/work/notes.md").read_text()
print("\nEARLY IDENTIFICATION NOTES")
print(notes281[:14000])

print("\nLATER EFFECT / TARGET EVIDENCE")
sections281 = re.split(r"(?m)(?=^## )", notes281)
matches281 = [
    s for s in sections281
    if re.search(r"identif|causal|target|true effect|ground.truth|instrument|estimated effect|record.based", s, re.I)
]
for s in matches281[-14:]:
    print(s[:1800])

print("\nLATEST FORWARD FILES AND VALIDATION IDS")
for k in (29, 30, 31, 32):
    p = Path(f"/work/fwd_{k}")
    if not p.exists():
        continue
    print(k, [q.name for q in p.iterdir()])
    for q in sorted(p.glob("*.csv")):
        df = pd.read_csv(q)
        print(q.name, df.shape, df.columns.tolist(), df.head(2).to_dict("records"))
        idcols = [c for c in df.columns if "stay" in c.lower() and "hour" not in c.lower()]
        if idcols:
            print("IDS", df[idcols[0]].drop_duplicates().tolist())

print("\nRELEVANT LIVE HELPERS")
keys281 = list(globals())
for key in keys281:
    obj = globals()[key]
    if callable(obj) and re.search(r"candidate|sym|asym|gate|factual_eval194|atom_clean58|rp_linear|rp_mlp", key):
        try:
            print(key, inspect.signature(obj))
        except Exception:
            pass

print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 281\nRecovered architecture, record-based identification notes, latest validation IDs and helper signatures. No weights changed. Run-280 asymmetric insulin candidate still needs GPU factual and rollout validation.\n")
```
