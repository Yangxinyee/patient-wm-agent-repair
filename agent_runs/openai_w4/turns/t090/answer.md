The records support an immediate negative insulin effect and positive norepinephrine effect. The estimated lag kernels also show persistence, so a one-step repair is only a starting point. I’ll now identify the training and independent validation arrays and measure the trade-off between restoring dose sensitivity and preserving factual NLL.

```python
import numpy as np
import torch
from pathlib import Path

print("=== HEAD DATASETS ===")
for _name, _obj in list(globals().items()):
    if hasattr(_obj, "columns") and {"prev", "observed_bin"}.issubset(_obj.columns):
        print(_name, _obj.shape, list(_obj.columns))
        print(_obj.head(2).to_string(index=False))

print("\n=== STATE BANKS ===")
for _name, _obj in list(globals().items()):
    if isinstance(_obj, dict) and {"x", "stay_id", "hour"}.issubset(_obj):
        print(_name, np.asarray(_obj["x"]).shape,
              "stays", len(np.unique(_obj["stay_id"])),
              "first IDs", np.unique(_obj["stay_id"])[:3])

print("\n=== HEAD ARRAYS ===")
for _name, _obj in list(globals().items()):
    if not _name.startswith("_") and isinstance(_obj, (np.ndarray, torch.Tensor)):
        if len(_obj.shape) >= 2 and (
            _obj.shape[0] in [4699, 4903] or
            any(_s in _name.lower() for _s in ["phi", "feat", "head"])
        ):
            print(_name, tuple(_obj.shape))

print("\n=== HEAD CACHE ===")
print("head_W_repair value tensors:",
      [k for k in head_W_repair if "attr_heads.value" in k])
print("repair_records:", repair_records.shape)
print("repair_bank keys:", list(repair_bank))
print("edited:", edited())

print("\n=== EXISTING PROBE TRANSFER ===")
for _key, _obj in head_innovation_probes.items():
    print(_key, {k: _obj[k] for k in ["r2", "rows"] if k in _obj})

print("\n=== FORWARD FILES ===")
for _dir in sorted(Path("/work").glob("fwd_*")):
    print(_dir.name, [p.name for p in _dir.iterdir()])

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Run 79\n"
        "Record-based immediate effects remain insulin -2.049/rate and NE "
        "+3.171/rate; deconvolved kernels show additional persistence. "
        "Preparing a closed-form head-feature regression with independent "
        "factual validation, not treating one-step fit as rollout success. No edits.\n"
    )
```
