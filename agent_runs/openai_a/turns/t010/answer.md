The architecture separates drug inputs from readings, so a targeted edit may be possible. But the records still need to distinguish drug effects from dosing-policy correlations. I’ll inspect the retained data structures and relevant weights; this will also show whether we can build a correction that stays nearly silent on factual records.

```python
import numpy as np
import pandas as pd
import json

print("=== KERNEL NAMES ===")
print([k for k in globals() if not k.startswith("_")])

print("\n=== DATAFRAMES ===")
for k, v in list(globals().items()):
    if isinstance(v, pd.DataFrame):
        print(k, v.shape, list(v.columns))

print("\n=== VOCAB STRUCTURE ===")
print(list(VOCAB.keys()))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))
for k in VOCAB:
    if any(s in k.lower() for s in ["norm", "unit", "categor", "kind"]):
        print(k, str(VOCAB[k])[:1800])

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in [
        "embed.numeric", "encode_action", "z_post", "z_prior", "z_head",
        "query_norm", "last_numeric", "type_emb", "attr_heads.value_bin",
        "embed_transition.value_bin", "out_norm", "core.gap"
    ]):
        print(name, SHAPES[name])

print("\n=== COMPLETE SENSOR-INSTRUMENT NOTE ===")
nt = open("/work/notes.md").read()
start = nt.find("## Original rollout")
end = nt.find("## Factual baseline", start)
print(nt[start:end][:4500])

print("\n=== FORWARD OUTPUT FILES ===")
from pathlib import Path
for folder in ["/work/fwd_1", "/work/fwd_2"]:
    print(folder, [(p.name, p.stat().st_size) for p in Path(folder).iterdir()])
```