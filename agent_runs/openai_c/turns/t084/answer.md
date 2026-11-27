The record-based estimates imply persistent effects: roughly −12.5/+6.3 for insulin and +4.7/−2.3 for norepinephrine on the training population. My assumed rollout filenames were wrong, so I have not checked the model’s persistence yet. I’ll locate the saved results and examine both the time courses and the reading changes caused by the numeric edit.

```python
import numpy as np, pandas as pd
from pathlib import Path

print("=== SAVED FORWARD FILES ===")
forward_files = {}
for directory in sorted(Path("/work").glob("fwd_*"), key=lambda p: int(p.name.split("_")[1])):
    names = [p.name for p in directory.iterdir()]
    forward_files[int(directory.name.split("_")[1])] = names
    print(directory.name, ", ".join(names))

print("\n=== ORIGINAL ROLLOUT TIME COURSE ===")
rr = original_rollout.copy()
if {"hour", "edit", "shift"}.issubset(rr.columns):
    print(rr.loc[rr.hour.between(13,24)].groupby(["hour","edit"])["shift"]
          .mean().unstack().round(3).to_string())

print("\n=== EARLIER PATHWAY TESTS ===")
text = Path("/work/notes.md").read_text()
start = text.find("## Runs 9")
end = text.find("## Python run 23")
print(text[start:end][:4400])

print("\n=== READING / STATE ALIGNMENT ===")
print("baseline factual", baseline_factual.head(3).to_dict("records"))
print("state keys", list(state24_original))
print("first state metadata",
      {k: state24_original[k][:6].tolist()
       for k in ["stay_id", "hour", "has_insulin", "has_norepinephrine"]})
print("probe ids", list(probe_ids)[:3], "count", len(probe_ids))
print("sensor/output sizes",
      {n: getattr(globals().get(n), "shape", None)
       for n in ["D_h","G_h","R_sensor","T_output","pilot_basis","pilot_design"]})
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 58\n"
            "- Recovered training-population targets: insulin -12.515/+6.257; NE +4.673/-2.336.\n"
            "- Persistence estimates: insulin rho 0.841, NE rho 0.673.\n"
            "- Locating saved rollout files and aligning paired states with factual predictions.\n"
            "- No edits changed; forward passes used: 24.\n")
```