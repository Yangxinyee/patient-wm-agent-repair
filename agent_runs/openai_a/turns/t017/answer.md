The architecture exposes two possible repair points: how drug doses enter the recurrent state, and how that state produces reading probabilities. The original rollouts show small insulin shifts and negative norepinephrine shifts, but those alone cannot tell us which needs correction. I’ll recover the remaining causal checks and compact state metadata to choose a targeted experiment.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

nt = Path("/work/notes.md").read_text()
print("Notebook length:", len(nt))
print("\n=== IDENTIFICATION NOTES, MIDDLE ===")
print(nt[3500:8500])

print("\n=== ARRAY / DATAFRAME VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, dict) and k not in {"SHAPES", "VOCAB", "__builtins__"}:
        print(k, "dict", len(v), str(list(v)[:8])[:140])

print("\n=== SELECTED WEIGHT SHAPES ===")
for n in NAMES:
    if any(s in n for s in ["attr_heads.value", "last_numeric", "query_norm",
                             "type_emb", "z_head", "out_norm",
                             "encode_action", "embed.numeric", "z_post"]):
        print(n, SHAPES[n])

print("\n=== FORWARD FILE SCHEMAS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    for p in sorted(folder.iterdir()):
        if p.suffix == ".csv":
            df_check = pd.read_csv(p)
            print(str(p), df_check.shape, list(df_check.columns))
            print(df_check.head(2).to_string(index=False))
        elif p.suffix == ".npz":
            zz = np.load(p)
            print(str(p), {k: zz[k].shape for k in zz.files})
```
