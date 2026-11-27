The architecture confirms that dose changes affect recurrent state and that generated readings feed back during rollout. Immediate slopes therefore aren’t enough. The output was truncated before the audit IDs and later causal findings, so I’ll retrieve those compactly. The candidate remains unvalidated until its GPU factual error is within 0.005.

```python
from pathlib import Path
import pandas as pd
import re

print("=== INDEPENDENT AUDIT IDS ===")
audit272_ids = None
for k in [19, 22]:
    folder = Path(f"/work/fwd_{k}")
    print(folder.name, [p.name for p in folder.iterdir()] if folder.exists() else [])
    for p in sorted(folder.glob("factual*.csv")):
        d = pd.read_csv(p)
        print(p.name, d.shape, list(d.columns))
        idcol = next((c for c in d.columns if c.lower() in ("stay", "stay_id", "stayid")), None)
        if idcol:
            ids = d[idcol].drop_duplicates().tolist()
            if audit272_ids is None or k == 22:
                audit272_ids = ids
            if "original" in p.name:
                print("STAYS", ids)
        for c in d.columns:
            if re.search(r"nll|loss", c, re.I) and pd.api.types.is_numeric_dtype(d[c]):
                print(c, float(d[c].mean()))

print("\n=== LATER RECORD-BASED CONCLUSIONS ===")
text = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)(?=^## )", text)
budget = 4800
for section in sections:
    header = section.splitlines()[0] if section else ""
    nums = re.findall(r"\d+", header)
    run = int(nums[0]) if nums else -1
    if 12 <= run <= 100 and re.search(
        r"causal|identif|threshold|record-derived|persistence|effect estimate|rollout target",
        section, re.I
    ):
        excerpt = section[:min(750, budget)]
        print(excerpt)
        budget -= len(excerpt)
        if budget <= 0:
            break

print("\n=== RELEVANT LIVE OBJECTS ===")
for name in sorted(list(globals())):
    if re.search(r"^(arch263|grid269|backup267|trial263|score263)$|target.*(effect|roll)|^(rho|beta)|truth|causal", name, re.I):
        obj = globals()[name]
        print(name, type(obj).__name__, getattr(obj, "shape", None),
              list(obj.keys())[:12] if isinstance(obj, dict) else
              str(obj)[:160] if isinstance(obj, (float, int, str, tuple, list)) else "")
print("AUDIT_IDS_READY", audit272_ids)
```
