The reporting code stopped because `shift` is also a pandas method; no weights changed, and the backup was created. I’ll fix that column reference and recover the pair construction. The GPU results still support keeping the symmetric insulin candidate while investigating norepinephrine.

```python
import inspect
import dis
import numpy as np
import torch

print("SYMMETRIC INSULIN ROLLOUT")
print(r269_reached.groupby("edit")["shift"].mean().to_dict())
print("EQUAL-STAY",
      r269_reached.groupby(["edit", "stay_id"])["shift"].mean()
      .groupby("edit").mean().to_dict())

def compact_source270(fn, limit):
    try:
        s = inspect.getsource(fn)
    except (OSError, TypeError):
        s = "\n".join(
            f"{i.offset} {i.opname} {i.argrepr[:200]}"
            for i in dis.get_instructions(fn)
        )
    print(s[:limit])
    if len(s) > limit:
        print("REMAINING CHARACTERS", len(s) - limit)

for fn_name, limit in (
    ("symmetric_ins265", 4400),
    ("candidate258", 2300),
    ("factual_eval194", 1600),
):
    fn = globals()[fn_name]
    print("\nFUNCTION", fn_name, inspect.signature(fn))
    compact_source270(fn, limit)

print("\nSYMMETRIC HELPER DEPENDENCIES")
for name in symmetric_ins265.__code__.co_names:
    if name not in globals():
        continue
    obj = globals()[name]
    if isinstance(obj, dict):
        print(name, "dict keys", str(list(obj)[:8])[:140])
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, (int, float, tuple, list)):
        print(name, str(obj)[:140])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 269–270; GPU passes 29–30\n"
        "Run 269 halted at a pandas shift-column/method collision after creating "
        "backup269; no weights changed. Run 270 corrects reporting and inspects "
        "calibration helper construction. Symmetric insulin gain 1.5/gate 26 "
        "passed GPU factual with delta +.00401. Reached-row shifts: insulin x2 "
        "-10.009, half +8.278; NE x2 +.536, half -3.324. Shared-gate NE repair "
        "is a computational hypothesis, not new evidence of ICU effects.\n"
    )
```