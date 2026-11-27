The current edit substantially improved all four shifts, but insulin’s response was smaller on these untouched stays than on the earlier cohort. I’ll compare it with targets at exactly the same measured hours and separate the two edit components. If gaps remain, a controlled gain trial may help; if the matched targets already agree, strengthening would be unjustified.

```python
import numpy as np
import pandas as pd
import re
from pathlib import Path

rollout218 = {}
for m218 in ["original", "edited"]:
    d218 = pd.read_csv(f"/work/fwd_28/dose_edit_{m218}.csv")
    j218 = d218.merge(record_targets217, on=["stay_id", "hour", "edit"],
                     how="left", validate="many_to_one")
    assert j218.target.notna().all()
    j218 = j218.loc[j218.reached.astype(bool)].copy()
    rollout218[m218] = j218
    print("\nMATCHED UNTOUCHED ROLLOUT", m218)
    print(j218.groupby("edit")[["shift", "target"]].mean().round(6).to_string())
    print("equal-stay",
          j218.groupby(["edit", "stay_id"])[["shift", "target"]].mean()
              .groupby("edit").mean().round(6).to_dict())

# Recover the exact two rank-one components, without changing weights.
V218 = np.asarray(directions212, dtype=np.float64)
W218 = np.asarray(candidate_W212, dtype=np.float64)
base218 = original_B215.numpy().astype(np.float64)
current218 = current_B215.numpy().astype(np.float64)
rank_components218 = [
    np.outer(V218[:, j], W218[:, j]) for j in range(2)
]
basis218 = np.column_stack([x.ravel() for x in rank_components218])
gains218 = np.linalg.lstsq(basis218, (current218-base218).ravel(), rcond=None)[0]
delta_ins218 = gains218[0] * rank_components218[0]
delta_ne218 = gains218[1] * rank_components218[1]
factor_error218 = np.max(np.abs(base218 + delta_ins218 + delta_ne218 - current218))
print("\nEXACT EDIT FACTORIZATION")
print("gains", gains218, "maximum reconstruction error", factor_error218)
print("base_B212 vs original", np.max(np.abs(np.asarray(base_B212)-base218)))
assert factor_error218 < 2e-5

print("\nRECORD EFFECT OBJECTS")
for n218 in ["effects", "effect"]:
    if n218 in globals():
        print(n218, repr(compact216(globals()[n218]))[:1000])
audit_names218 = [
    n for n, o in list(globals().items())
    if re.search(r"181|182|183", n) and not callable(o) and not n.startswith("_")
]
print("audit object names", ", ".join(audit_names218)[:1200])

notes218 = Path("/work/notes.md").read_text()
sections218 = notes218.split("\n## ")
evidence_sections218 = [
    s for s in sections218
    if re.search(r"IV|cross.fit|identif|sensor.history|sensor/history", s, re.I)
    and not re.search(r"Run21[4-8]", s[:100])
]
print("\nCOMPACT IDENTIFICATION NOTES")
for s218 in evidence_sections218[-3:]:
    print("##", s218[:1100])

with open("/work/notes.md", "a") as f218:
    f218.write(
        "\n## Forward28 / run218: untouched rollout and exact factor recovery\n"
        "Current candidate improves all four responses over original on untouched stays. "
        "Compared responses to record targets at exactly the reached measurement hours. "
        f"Recovered gains {gains218.tolist()} and separate delta_ins218/delta_ne218, "
        f"factor error {factor_error218:.3g}. No weight changes. "
        "Any strengthening requires direct factual validation.\n"
    )
```
