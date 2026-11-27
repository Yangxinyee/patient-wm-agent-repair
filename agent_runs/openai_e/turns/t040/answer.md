The retained data include the causal estimates, factual baseline and GPU states, but the output was again truncated before the action-path objects. I’ll narrow the inspection to those objects and tensor dimensions. That will show whether a dose-sensitive latent correction can be computed and tested directly.

```python
import numpy as np
import pandas as pd
import torch

print("=== COMPACT ARRAY INVENTORY ===")
items = []
for key, val in list(globals().items()):
    if key.startswith("_") or key in {"val", "obj", "xx", "X", "y"}:
        continue
    if isinstance(val, np.ndarray) and val.size > 100:
        items.append(f"{key}:{val.shape}")
    elif isinstance(val, torch.Tensor) and val.numel() > 100:
        items.append(f"{key}:{tuple(val.shape)}")
print("; ".join(items))

print("=== DICTIONARY NAMES ===")
for key, val in list(globals().items()):
    if key.startswith("_") or key in {"SHAPES", "VOCAB", "obj", "val"}:
        continue
    if isinstance(val, dict) and val:
        print(key, len(val), str(list(val.keys())[:5])[:120])

print("=== RELEVANT TENSOR NAMES ===")
for key in NAMES:
    if any(s in key for s in ["encode_action", "z_post", "z_in", "z_head",
                               "attr_heads.value", "embed.numeric"]):
        print(key, SHAPES[key])

print("=== RECORD DEFINITIONS ===")
print(Path("/data/README.md").read_text())
print("=== CAUSAL RESULTS ===")
print(rd_table.to_string(index=False))
print("=== ORIGINAL ROLLOUT SUMMARY ===")
print(baseline_rollout.groupby("edit").apply(
    lambda g: pd.Series({"shift": g.loc[g.reached.astype(bool), "shift"].mean(),
                        "reached_hours": int(g.reached.sum())}),
    include_groups=False
).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 37: targeted action-path inventory\n"
            "Full panels, causal models, original factual results and GPU states persist. Narrowing recovery to action/latent features and exact tensor names so an actual reversible edit can be tested. No edits. Python runs: 37; forward passes: 3.\n")
```