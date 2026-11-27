The tensor shapes confirm that small, low-rank edits are feasible. The output was truncated before the state-file details, so I’ll retrieve just those essentials now. The next check is whether dose information can be decoded from the state well enough to distinguish recorded treatment from an intervention.

```python
from pathlib import Path
import numpy as np
import torch

print("=== POSTERIOR, PRIOR, AND HEAD TENSORS ===")
for name in NAMES:
    if name.startswith(("z_", "gap.", "query_norm.", "type_emb.",
                        "filter_", "pending_", "last_", "attr_heads.value_bin.")):
        print(name, SHAPES[name])

print("\n=== NUMERIC VOCAB ===")
print("numeric structure:", type(VOCAB["numeric"]).__name__)
print(repr(VOCAB["numeric"])[:3500])

print("\n=== FORWARD ARTIFACTS / STATE SCHEMA ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
for path in sorted(Path("/work").glob("fwd_*/*states*.npz")):
    tmp35 = np.load(path)
    print(str(path), [(k, tmp35[k].shape, str(tmp35[k].dtype))
                     for k in tmp35.files])
    for key in tmp35.files:
        if any(s in key.lower() for s in ("stay", "hour", "insulin", "norep")):
            print(key, tmp35[key][:8])

print("\n=== KEY RETAINED OBJECTS ===")
for name in ("P", "A", "train", "tuning", "df", "ds22", "d22",
             "r32_base", "r32_dt", "r32_target", "kernels19", "kernel_se19"):
    val = globals().get(name)
    if val is None:
        continue
    if isinstance(val, dict):
        print(name, "keys:", list(val)[:15])
    elif hasattr(val, "columns"):
        print(name, val.shape, "columns:", list(val.columns))
        if name in ("ds22", "d22"):
            print(val.head(3).to_string(index=False))
    elif hasattr(val, "shape"):
        print(name, val.shape)
    else:
        print(name, repr(val)[:400])

print("\n=== DECODER FEATURE RETURN ===")
features35 = r32_features(r32_best)
print("function names:", r32_features.__code__.co_names)
print("constants:", r32_features.__code__.co_consts)
if isinstance(features35, (tuple, list)):
    print([(type(v).__name__, getattr(v, "shape", None)) for v in features35])
else:
    print(type(features35).__name__, getattr(features35, "shape", None))

print("\n=== DATA VARIABLE NAMES ===")
print([k for k in sorted(globals()) if
       k.startswith(("st", "tr", "tun", "out22", "r32_"))
       and not k.startswith("__")])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 35\n"
            "Run 34 output was truncated by function disassembly. Retrieving "
            "compact posterior/head tensor names, normalization, state schema "
            "and retained feature returns. No tensors edited; forward 3/50.\n")
```