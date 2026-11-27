Independent record-based estimates consistently suggest a negative insulin effect and a positive norepinephrine effect. Their accumulated effects remain uncertain because persistence is harder to identify. I’ll inspect the available head features and data alignment next; a misaligned state or register could explain the reconstruction errors and invalidate an edit.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print("=== STATE FILE ===")
print(type(S).__name__, list(S.keys()))
for key in S.keys():
    a = S[key]
    print(key, a.shape, a.dtype, str(a[:2])[:250])

print("\n=== HEAD ALIGNMENT ===")
print(head_data.columns.tolist())
print(head_data.head(3).to_string(index=False))
print(state_meta.head(3).to_string(index=False))

print("\n=== HEAD FUNCTION INTERNALS ===")
print("variables:", local_value_head.__code__.co_varnames)
print("names:", local_value_head.__code__.co_names)
print("constants:", [c for c in local_value_head.__code__.co_consts
                     if isinstance(c, (str, int, float, tuple))])
print("weight keys:", list(head_weights))

print("\n=== RETURNED FEATURES ===")
probe_features = local_value_head(
    head_data.iloc[:3], head_cfg, return_features=True
)
if isinstance(probe_features, dict):
    for key, value in probe_features.items():
        print(key, type(value).__name__, getattr(value, "shape", ""),
              str(value)[:150] if not hasattr(value, "shape") else "")
else:
    print(type(probe_features).__name__, str(probe_features)[:800])

print("\n=== NUMERIC NORMALISATION ===")
for key, value in numeric_specs.items():
    print(key, {k: v for k, v in value.items() if k != "fine_bin_edges"})

print("\n=== PANEL AND EXPOSURE SUMMARY ===")
print(panel.columns.tolist())
print(panel.head(3).to_string())
for drug in ["insulin", "norepinephrine"]:
    rates = panel[drug]
    print(drug, "positive rate quantiles:",
          rates[rates > 0].quantile([0, .25, .5, .75, .95, 1]).to_dict())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 22: consolidated evidence and alignment audit\n"
        "Insulin clean lab threshold estimates approximately -1.55/rate unit; "
        "history-adjusted lab estimates approximately -1.97. NE rich-history "
        "estimates approximately +3.5 to +3.9/rate unit. Positive persistence "
        "is supported, but accumulated rollout targets remain uncertain. "
        "No edits; inspecting state/head alignment and exact features.\n"
    )
```