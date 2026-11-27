The architecture mixes actions and observations in its recurrent state, and the value head also uses last-reading shortcuts. There is no explicit drug-effect coefficient to replace. I’ll inspect the head shapes, saved states, and factual baseline to see whether a small residual-dose correction is feasible; it needs to affect changed doses much more than ordinary records.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== KERNEL VARIABLE NAMES ===")
print(sorted(n for n in globals() if not n.startswith("_")))

print("\n=== DATAFRAME INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, "columns:", list(obj.columns)[:55])

print("\n=== STATE / FEATURE OBJECTS ===")
for name in ("S", "F", "P", "T", "train", "tuning", "panel", "tpanel", "state_ids"):
    obj = globals().get(name)
    if obj is None:
        continue
    print(name, type(obj).__name__, getattr(obj, "shape", None))
    if isinstance(obj, dict):
        print(" keys:", list(obj.keys()))
        for k, v in list(obj.items())[:16]:
            print(" ", k, getattr(v, "shape", None), str(getattr(v, "dtype", "")))

print("\n=== TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith(("attr_heads.", "embed.", "encode_action.", "core.z_", "core.zhead",
                        "core.z_head", "core.out_norm", "query", "last_", "type_emb", "gap",
                        "embed_transition.")):
        print(name, SHAPES[name])

print("\n=== VOCAB SUMMARY ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for key in ("normalisation", "normalization", "numeric_stats"):
    if key in VOCAB:
        print(key, VOCAB[key])

print("\n=== EXISTING FORWARD FILES ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [(p.name, p.stat().st_size) for p in directory.iterdir()])
    for p in directory.glob("*.json"):
        text = p.read_text()
        print(p.name, text[:1600])

print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())
```