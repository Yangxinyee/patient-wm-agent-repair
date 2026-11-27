The symmetric pair improved insulin rollouts: doubling gave −10.01 and halving +8.28, compared with record-derived row-weighted targets near −15.01 and +7.50. Norepinephrine remains poorly calibrated at +0.54 and −3.32. I’ll preserve this passing candidate and inspect the norepinephrine pair; a shared-gate version may improve its response, but must also pass factual testing.

```python
import numpy as np
import torch
import inspect
import dis
import io
import contextlib
import json
from pathlib import Path

backup269 = {
    n: (load(n).clone() if isinstance(load(n), torch.Tensor) else load(n).copy())
    for n in edited()
}

r269_roll = pd.read_csv("/work/fwd_31/dose_edit_edited.csv")
r269_reached = r269_roll[r269_roll.reached.astype(str).str.lower().eq("true")]
print("SYMMETRIC INSULIN ROLLOUT")
print(r269_reached.groupby("edit").shift.mean().to_dict())
print("EQUAL-STAY",
      r269_reached.groupby(["edit", "stay_id"])["shift"].mean()
      .groupby("edit").mean().to_dict())

print("\nCALIBRATION IMPLEMENTATIONS")
for fn_name in ("symmetric_ins265", "candidate258", "factual_eval194"):
    fn = globals()[fn_name]
    print("\n", fn_name, inspect.signature(fn))
    try:
        text = inspect.getsource(fn)
    except (OSError, TypeError):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dis.dis(fn)
        text = buf.getvalue()
    print(text[:4200 if fn_name == "symmetric_ins265" else 2400])

print("\nSYMMETRIC HELPER DEPENDENCIES")
for name in symmetric_ins265.__code__.co_names:
    if name not in globals():
        continue
    obj = globals()[name]
    if isinstance(obj, dict):
        print(name, "dict keys", str(list(obj)[:10])[:180])
    elif hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, (int, float, tuple, list)):
        print(name, str(obj)[:180])

r269_note = (
    "\n## Run 269; GPU passes 29–30\n"
    "Run-266 symmetric insulin gain 1.5/gate 26 passed GPU factual: "
    "1.59753 versus 1.59352, delta +.00401. Reached-row dose shifts: "
    "insulin x2 -10.009, half +8.278; NE x2 +.536, half -3.324. "
    "Insulin improved substantially, especially halving; doubling remains "
    "below the record-derived target. Preserved all saved tensors in backup269. "
    "Inspecting NE shared-gate symmetry without changing submitted weights.\n"
)
with open("/work/notes.md", "a") as f:
    f.write(r269_note)
```