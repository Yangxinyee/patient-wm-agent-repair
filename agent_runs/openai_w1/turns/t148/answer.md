The new export provides complete states for two training stays, but it does not yet validate our replay or any repair. I’ll recover the architecture, earlier causal estimates, and available helpers before editing weights. Exact state agreement would validate replay; otherwise, we should rely on exported states and GPU checks.

```python
import os, json, inspect, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("=== RECORD SCHEMA ===")
print(open("/data/README.md").read())

resume_notes = open("/work/notes.md").read()
print("=== EARLY NOTEBOOK ===")
print(resume_notes[:18000])

print("=== LIVE FUNCTIONS ===")
for resume_name, resume_obj in sorted(list(globals().items())):
    if resume_name.startswith("_") or resume_name.startswith("resume_"):
        continue
    if inspect.isfunction(resume_obj):
        try:
            resume_sig = str(inspect.signature(resume_obj))
        except Exception:
            resume_sig = "?"
        print(resume_name, resume_sig)

print("=== LIVE DATA OBJECTS ===")
for resume_name, resume_obj in sorted(list(globals().items())):
    if resume_name.startswith("_") or resume_name.startswith("resume_"):
        continue
    if isinstance(resume_obj, pd.DataFrame):
        print(resume_name, "DataFrame", resume_obj.shape,
              list(resume_obj.columns[:18]))
    elif isinstance(resume_obj, (np.ndarray, torch.Tensor)):
        print(resume_name, type(resume_obj).__name__,
              tuple(resume_obj.shape), str(resume_obj.dtype))
    elif isinstance(resume_obj, dict) and len(resume_obj) < 100:
        print(resume_name, "dict keys", list(resume_obj.keys())[:30])

resume_s7 = np.load("/work/fwd_7/states_original.npz")
print("=== NEW EXPORT ===")
for resume_key in resume_s7.files:
    resume_a = resume_s7[resume_key]
    print(resume_key, resume_a.shape, resume_a.dtype)
print("stay ids:", np.unique(resume_s7["stay_id"]))
print("edited tensors:", edited())

with open("/work/notes.md", "a") as resume_f:
    resume_f.write(
        "\n## Run 135\n"
        "Restart recovery: reading architecture, original record evidence, "
        "and live helper inventory. Forward 7 supplies 99 complete bundle "
        "states for two training stays; it is not itself replay validation. "
        "No weights edited. Counts: 135 Python runs, 7 forward passes.\n"
    )
```