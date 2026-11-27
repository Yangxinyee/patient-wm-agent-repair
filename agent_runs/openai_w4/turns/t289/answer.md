The intermediate gate restored insulin responses to −7.539 and +3.566, close to a two-to-one ratio, while retaining factual fit. Norepinephrine still looks poorly calibrated. Before changing it, I’ll verify the targets’ dose units and reached-hour weighting. That will determine whether insulin is already close or still too weak.

```python
import re, types
import numpy as np
import torch
from pathlib import Path

r250_passing = {n: torch.as_tensor(load(n)).clone() for n in edited()}

print("MATCHED TARGET AGGREGATIONS:")
for edit_name, obj in matched_targets238.items():
    print(edit_name, "keys:", list(obj.keys()))
    frame = obj.get("frame")
    if frame is not None and "kernel_shift_raw_rate" in frame:
        vals = frame["kernel_shift_raw_rate"]
        print(" raw-rate reached-row mean:", float(vals.mean()),
              "equal-stay mean:",
              float(frame.groupby("stay_id")["kernel_shift_raw_rate"].mean().mean()),
              "rows:", len(frame))
    for key, val in obj.items():
        if isinstance(val, (int, float, np.number)):
            print(" ", key, float(val))

print("IDENTIFICATION NOTE EXCERPTS:")
text250 = Path("/work/notes.md").read_text()
for section in re.split(r"(?m)(?=^#{1,3} )", text250):
    header = section.split("\n", 1)[0]
    if re.search(r"\b(136|137|138|185|186|187)\b", header):
        print(section[:1050])

print("KEY FUNCTION METADATA:")
for name in ("fit_lp136", "deconvolve_lp136", "gated_ne185",
             "collect_channel186", "collect_selected187", "factual_eval194"):
    fn250 = globals()[name]
    c250 = fn250.__code__
    print(name, "args:", c250.co_varnames[:c250.co_argcount],
          "globals:", c250.co_names)

print("IDENTIFICATION / NE OBJECTS:")
for name, obj in list(globals().items()):
    if not name.startswith("_") and (
        re.search(r"(136|137)$", name) or
        any(s in name for s in ("ne185", "ne186", "ne187"))
    ):
        if isinstance(obj, types.FunctionType):
            continue
        if hasattr(obj, "shape"):
            arr = np.asarray(obj)
            desc = (arr.round(6).tolist() if arr.size <= 20
                    else ("shape", arr.shape))
        elif isinstance(obj, dict):
            desc = ("keys", list(obj.keys())[:16])
        elif isinstance(obj, (str, int, float, list, tuple)):
            desc = repr(obj)[:180]
        else:
            desc = type(obj).__name__
        print(name, desc)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 250; GPU passes 25–26\n"
            "Gain-1.25/gate-28 candidate (hash 29031359da) passed GPU factual: "
            "1.59668 vs 1.59352, delta +.00316 on 98 stays. Reached-row shifts: "
            "insulin x2 -7.539, half +3.566; NE x2 +.705, half -3.260. "
            "Insulin asymmetry is much reduced; NE calibration remains problematic. "
            "Saved passing backup r250_passing. Auditing longitudinal-kernel dose units "
            "and aggregation before accepting calibration targets. No weights changed.\n")
```
