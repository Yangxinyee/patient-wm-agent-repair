The stronger head increased factual NLL by **0.00560**, exceeding the allowed margin on this cohort. I’ll restore the previously audited candidate and recover the records-based effect estimates before choosing a final edit. Further independent factual checks must remain below +0.005; improved dose responses alone are not enough.

```python
from pathlib import Path
import numpy as np
import torch
import re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

backup292_strong = {n: torch.as_tensor(load(n)).clone() for n in edited()}
assert isinstance(backup290, dict)
assert all(n in backup290 for n in edited())
for n in list(edited()):
    save_edit(n, backup290[n])
print("RESTORED:", sorted(edited()))

notes292 = Path("/work/notes.md").read_text()
lines292 = notes292.splitlines()
hits292 = [
    i for i, line in enumerate(lines292)
    if re.search(r"DML|threshold|accepted.*effect|corrected.*effect|causal.*estimate|instrument|true.*effect",
                 line, re.I)
]
selected292 = set()
for i in hits292:
    selected292.update(range(max(0, i-1), min(len(lines292), i+4)))
excerpt292 = "\n".join(lines292[i] for i in sorted(selected292))
print("=== RECORDS-BASED IDENTIFICATION NOTES ===")
print(excerpt292[-14000:])

def describe292(obj, depth=0):
    if isinstance(obj, dict):
        print("dict keys:", list(obj)[:35], "; total:", len(obj))
        if depth < 1:
            for k in list(obj)[:2]:
                print(" key:", k)
                describe292(obj[k], depth+1)
    elif isinstance(obj, (tuple, list)):
        print(type(obj).__name__, "length:", len(obj))
        for j, x in enumerate(obj[:12]):
            print(j, type(x).__name__, getattr(x, "shape", None))
            if np.isscalar(x):
                print(" value:", x)
    else:
        print(type(obj).__name__, getattr(obj, "shape", None))

for key292 in ["grid286", "weighted281", "backup287", "backup290"]:
    print("=== OBJECT", key292, "===")
    describe292(globals()[key292])

print("=== SELECTED GRID286 CANDIDATE ===")
describe292(grid286[(.35, .3, 8)], depth=0)

print("=== AVAILABLE FIT/SCORING HELPERS ===")
for key292 in sorted(list(globals())):
    if re.search(r"weighted_fit|nll228|slope|target|effect|quiet|audit|stay.*(id|list)", key292, re.I):
        obj292 = globals()[key292]
        print(key292, type(obj292).__name__, getattr(obj292, "shape", None))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 292 / factual-budget rollback\n"
        "Forward32 (fwd_34), strong grid286(.45,.4,8), factual delta "
        "+.00560 on tuning100–199, exceeds +.005 on this cohort. "
        "Preserved strong tensors in backup292_strong and restored all seven "
        "backup290 tensors: audited grid286(.35,.3,8). Recovering records-based "
        "targets before final independent factual/rollout audits.\n"
    )
```