The saved insulin baseline was deliberately retained, while several norepinephrine candidates failed validation. The notes give a working NE target of about +4.2 to +4.6 for doubling, but not a verified truth. I’ll compare the actual forward results to identify the best prior trade-off; that will test whether a smaller, better-anchored NE edit is worth pursuing.

```python
import json, pandas as pd, numpy as np, re
from pathlib import Path

print("FACTUAL VALIDATION TABLE")
for d187 in sorted(Path("/work").glob("fwd_*"),
                   key=lambda p: int(p.name.split("_")[1])):
    po187, pe187 = d187/"factual_original.csv", d187/"factual_edited.csv"
    if not (po187.exists() and pe187.exists()):
        continue
    o187, e187 = pd.read_csv(po187), pd.read_csv(pe187)
    nc187 = [c for c in o187.columns if "nll" in c.lower()]
    if nc187:
        c187 = nc187[0]
        print(d187.name, "n=", len(o187), "column=", c187,
              "original=", round(float(o187[c187].mean()), 6),
              "edited=", round(float(e187[c187].mean()), 6),
              "change=", round(float(e187[c187].mean()-o187[c187].mean()), 6))

print("\nDOSE SUMMARIES")
for n187 in [1, 5, 7, 10, 12, 15, 18]:
    p187 = Path(f"/work/fwd_{n187}/summary.json")
    if p187.exists():
        print("fwd_"+str(n187), json.dumps(json.loads(p187.read_text()))[:650])

print("\nCSV SCHEMAS")
for p187 in [Path("/work/fwd_21/factual_edited.csv"),
             Path("/work/fwd_18/dose_edit_edited.csv")]:
    x187 = pd.read_csv(p187, nrows=1)
    print(p187.name, list(x187.columns), x187.to_dict("records"))

print("\nLATE CANDIDATE DECISIONS")
ss187 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
for n187 in [70, 72, 76, 154, 155, 156, 157, 158, 159]:
    for s187 in ss187:
        if re.match(rf"## Run {n187}(?:\D|$)", s187):
            print(s187[:310].strip())

print("\nRETAINED MULTI-DIRECTION ARRAY SHAPES")
for k187, v187 in list(globals().items()):
    if re.search(r"(13[8-9]|14[0-7]|15[0-5])$", k187) and hasattr(v187, "shape"):
        print(k187, tuple(v187.shape))

with open("/work/notes.md", "a") as f187:
    f187.write("\n## Run 187: compare measured factual and intervention trade-offs\n"
               "Recovered forward-result tables and retained NE edit bases. "
               "No weights changed. Hypothesis to test: repaired-state anchoring can "
               "reduce NE factual leakage relative to original-state fits. "
               "Python runs: 187; forwards: 22.\n")
```