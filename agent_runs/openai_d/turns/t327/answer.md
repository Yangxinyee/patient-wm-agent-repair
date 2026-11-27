The second cohort confirms strong insulin responses (−13.562/+5.668) and correctly directed but weak norepinephrine responses (+1.378/−0.808). Insulin now looks substantially repaired; norepinephrine remains incomplete. I’ll inspect the saved strength comparisons and factual margins without changing weights. A stronger norepinephrine candidate is worth testing only if the evidence suggests it can retain a credible factual-fit margin.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import re

print("=== RECOVERED TARGET OBJECTS ===")
for _name in ("record_targets", "target_means", "rec_ins_target",
              "initial_effects", "trial_causal_beta"):
    if _name in globals():
        _v = globals()[_name]
        print(_name, type(_v).__name__, str(_v)[:1100])

print("=== RECENT REPAIR INTERFACES ===")
_keys272 = sorted(k for k in globals()
                  if re.match(r"(r26[2-9]|n262|r24[5-9]|r250|r25[5-8])", k))
for _name in _keys272:
    _v = globals()[_name]
    if callable(_v):
        _desc = "callable"
    elif isinstance(_v, dict):
        _desc = "keys=" + str(list(_v)[:16])
    elif hasattr(_v, "shape"):
        _desc = "shape=" + str(_v.shape)
    elif isinstance(_v, (list, tuple)):
        _desc = "length=" + str(len(_v))
    else:
        _desc = str(_v)[:120]
    print(_name, type(_v).__name__, _desc)

print("=== SELECTED LATE TRIAL CONCLUSIONS ===")
_sections272 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
_budget272 = 2600
for _s in _sections272:
    _h = _s.splitlines()[0] if _s.splitlines() else ""
    if re.search(r"\b(224|230|245|246|247|250|251|265)\b", _h):
        _piece = _s[:550]
        if len(_piece) <= _budget272:
            print(_piece)
            _budget272 -= len(_piece)

print("=== SAVED CHECKPOINT KEYS ===")
for _file in ("candidate90_35_bin80x4.npz",
              "rejected_8898bdc457.npz",
              "ne50_hidden_cancellation.npz",
              "lastbin80_scale4.npz"):
    with np.load("/work/" + _file) as _z:
        print(_file, [(k, _z[k].shape) for k in _z.files])

_roll272 = []
for _k in (43, 44):
    for _m in ("original", "edited"):
        _d = pd.read_csv(f"/work/fwd_{_k}/dose_edit_{_m}.csv")
        _d = _d[_d["reached"].astype(str).str.lower().isin(["true", "1"])]
        _d["model"] = _m
        _roll272.append(_d)
_roll272 = pd.concat(_roll272, ignore_index=True)
print("=== FIRST200 POOLED ROLLOUTS ===")
print(_roll272.groupby(["model", "edit"])["shift"]
      .agg(["count", "mean"]).to_string())

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Forward 44 / Run 272: second-cohort rollout consistency\n"
        "Checkpoint 864804d79e unchanged. Tuning100..199 edited shifts: "
        "insulin x2 -13.562, half +5.668; NE x2 +1.378, half -0.808. "
        "Strong insulin repair transfers; NE direction repaired but magnitude "
        "remains below provisional record-derived targets. Inspecting saved "
        "strength/cancellation interfaces before deciding whether another "
        "reversible trial is justified by the factual budget.\n"
    )
```
