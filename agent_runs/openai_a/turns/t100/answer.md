The earlier norepinephrine estimates change substantially across identification methods, so they do not yet justify an edit. For insulin, I’ll check whether the stronger graft’s effect reverses over time and recover its construction. A reversal would favor correcting the decoder’s off-policy behavior rather than increasing its coefficient.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import inspect
import re

rr_notes = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK SECTION INDEX ===")
for rr_m in re.finditer(r"^## .*$", rr_notes[:24000], re.M):
    print(rr_m.start(), rr_m.group())

print("\n=== THRESHOLD / RD EVIDENCE ===")
rr_hits = list(re.finditer(r"(?i)(regression discontinuity|threshold|insulin RD)", rr_notes))
for rr_m in rr_hits[:5]:
    print(rr_notes[max(0, rr_m.start()-120):rr_m.start()+850])

print("\n=== PAIRED ROLLOUT READOUTS (REACHED HOURS, THEN STAYS) ===")
rr_rollouts = {}
for rr_k in (9, 11):
    for rr_model in ("original", "edited"):
        rr_p = Path(f"/work/fwd_{rr_k}/dose_edit_{rr_model}.csv")
        if rr_p.exists():
            rr_d = pd.read_csv(rr_p)
            rr_d = rr_d[rr_d["reached"].astype(str).str.lower().isin(["true", "1"])]
            rr_rollouts[(rr_k, rr_model)] = rr_d
            rr_summary = rr_d.groupby(["edit", "stay_id"])["shift"].mean().groupby("edit").mean()
            print(rr_k, rr_model, rr_summary.round(4).to_dict())
print("\nInsulin x2: reached-hour means")
rr_hour_rows = {}
for rr_key, rr_d in rr_rollouts.items():
    rr_hour_rows[str(rr_key)] = (
        rr_d[rr_d.edit == "insulin_x2"].groupby("hour")["shift"].mean()
    )
print(pd.DataFrame(rr_hour_rows).round(3).to_string())

print("\n=== LOCAL CPU FUNCTIONS ===")
for rr_n, rr_o in sorted(list(globals().items())):
    if (inspect.isfunction(rr_o) and rr_o.__module__ == "__main__"
            and not rr_n.startswith("rr_")):
        print(rr_n, str(inspect.signature(rr_o))[:160])

print("\n=== GRAFT CONSTRUCTION DEPENDENCIES ===")
if "make_insulin_graft" in globals():
    print("globals:", make_insulin_graft.__code__.co_names)
    print("locals:", make_insulin_graft.__code__.co_varnames)
    try:
        print(inspect.getsource(make_insulin_graft)[:2500])
    except Exception:
        print("Source unavailable; dependencies above are retained.")
```