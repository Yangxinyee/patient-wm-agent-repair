The weaker broadened decoder has stronger immediate dose responses and lower offline factual error than the rejected candidate. It still needs an independent GPU audit. I’ll recover the audit stay IDs and the record-derived effect estimates before testing it; factual degradation above 0.005 would reject it.

```python
from pathlib import Path
import pandas as pd
import re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes271 = Path("/work/notes.md").read_text()
print("\n=== EARLY RECORD-BASED FINDINGS ===")
print(notes271[:13000])

print("\n=== NOTEBOOK IDENTIFICATION / EFFECT HEADINGS ===")
for m in re.finditer(r"(?m)^##.*$", notes271):
    section = notes271[m.start():min(len(notes271), m.start()+1600)]
    if re.search(r"instrument|randomiz|identif|true effect|causal estimate|record-derived|target effect", section, re.I):
        print(section[:1600])

print("\n=== RECENT FORWARD FILES AND STAY IDS ===")
for k in [19, 20, 21, 22]:
    folder = Path(f"/work/fwd_{k}")
    if not folder.exists():
        continue
    print(folder.name, [p.name for p in folder.iterdir()])
    for p in sorted(folder.glob("*.csv")):
        d = pd.read_csv(p)
        print(p.name, d.shape, list(d.columns))
        print(d.head(2).to_string(index=False))
        idcols = [c for c in d.columns if c.lower() in ("stay", "stay_id", "stayid")]
        if idcols:
            ids = d[idcols[0]].drop_duplicates().tolist()
            print("AUDIT_STAY_IDS", ids)
        numeric = d.select_dtypes(include="number")
        cols = [c for c in numeric.columns if re.search(r"nll|shift|loss", c, re.I)]
        if cols:
            print("MEANS", numeric[cols].mean().to_dict())

print("\n=== LIVE FIT INTERFACES ===")
for name in [
    "backup267", "backup268_rejected", "grid269", "arch263",
    "trial263", "score263", "score269", "T0", "GF", "GDI", "GDN",
    "RF", "RDI", "RDN", "feature_scale"
]:
    if name in globals():
        obj = globals()[name]
        print(name, type(obj).__name__,
              getattr(obj, "shape", None),
              list(obj.keys())[:20] if isinstance(obj, dict) else "")
print("CURRENT_EDITS", edited())
```
