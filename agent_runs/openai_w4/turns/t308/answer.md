The symmetric insulin edit passed the local factual screen with a +0.00370-nat change, but its GPU factual fit and dose responses remain untested. I’ll recover the validation stay IDs and retained helpers without changing weights. The next paired forward tests will show whether symmetry improves the dose responses while preserving factual fit.

```python
from pathlib import Path
import pandas as pd
import re

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

r267_snapshot = list(globals().items())
print("\nTUNING SCHEMA")
r267_cols = pd.read_csv("/data/tuning.csv", nrows=0).columns.tolist()
print(r267_cols)
r267_idcol = next((c for c in r267_cols if c.lower() in
                  ("stay_id", "stay", "patient_id", "episode_id")), r267_cols[0])
r267_tuning_ids = pd.read_csv(
    "/data/tuning.csv", usecols=[r267_idcol]
)[r267_idcol].drop_duplicates().tolist()
print("FIRST 98 TUNING IDS:", r267_tuning_ids[:98])

print("\nLATEST FORWARD FILES")
for d in ("fwd_25", "fwd_26", "fwd_27", "fwd_28", "fwd_29"):
    p = Path("/work") / d
    if p.exists():
        print(d, sorted(x.name for x in p.iterdir()))
        for f in sorted(p.glob("*.csv")):
            q = pd.read_csv(f)
            print(f.name, "shape", q.shape, "columns", q.columns.tolist())
            print(q.head(1).to_dict("records"))

print("\nRETAINED HELPERS AND BACKUPS")
r267_pat = re.compile(
    r"266|250|259|sym|pair|candidate|factual|replay|screen|backup|kernel|target",
    re.I
)
for name, obj in r267_snapshot:
    if not r267_pat.search(name):
        continue
    if callable(obj) and hasattr(obj, "__code__"):
        print("FUNCTION", name,
              "args", obj.__code__.co_varnames[:obj.__code__.co_argcount],
              "dependencies", obj.__code__.co_names[:35])
    elif isinstance(obj, dict):
        print("DICT", name, "keys", list(obj)[:20])
    elif hasattr(obj, "shape"):
        print("ARRAY", name, tuple(obj.shape))
    elif isinstance(obj, (str, int, float, tuple)):
        print("VALUE", name, str(obj)[:180])

print("\nSAVED EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 267\n"
            "Recovered architecture, tuning IDs, latest forward schemas and retained "
            "helpers before GPU validation of Run-266 symmetric insulin candidate. "
            "No weights changed. CPU factual delta remains +.00369549; GPU factual "
            "fit and rollout response are unvalidated.\n")
```