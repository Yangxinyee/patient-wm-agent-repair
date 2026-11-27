The architecture shows two relevant paths: drug doses update the recurrent state, while readings also use a same-code “last value” shortcut. A repair may need to strengthen drug response without disrupting that shortcut. I’ll recover the existing causal estimates and helper functions; stable history-adjusted dose effects would support a targeted repair, while unstable estimates would require more identification work.

```python
from pathlib import Path
import numpy as np, pandas as pd, torch, inspect

print("=== COMPACT RETAINED INVENTORY ===")
items = []
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"In", "Out", "NAMES", "SHAPES", "VOCAB", "items"}:
        continue
    if isinstance(v, pd.DataFrame):
        items.append(f"{k}: DF {v.shape} cols={list(v.columns)[:12]}")
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        items.append(f"{k}: {type(v).__name__} {tuple(v.shape)}")
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        items.append(f"{k}: function {inspect.signature(v)}")
    elif isinstance(v, dict):
        items.append(f"{k}: dict {list(v)[:10]}")
print("\n".join(items)[:4500])
print("\n=== REMAINING ACTION/LATENT TENSORS ===")
for n in NAMES:
    if n.startswith(("encode_action", "z_", "last_numeric", "embed.numeric", "embed.k", "pending", "filter")):
        print(n, SHAPES[n])
print("\n=== VOCAB STRUCTURE ===")
for k, v in VOCAB.items():
    if k in {"codes", "config"}:
        print(k, v)
    else:
        print(k, list(v)[:12] if isinstance(v, dict) else type(v).__name__)
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:2600])
print("\n=== BASELINE DOSE FILE ===")
p = next(Path("/work").glob("fwd_*/dose_edit_original.csv"), None)
if p:
    q = pd.read_csv(p)
    print(str(p), q.shape, list(q.columns))
    print(q.head(3).to_string(index=False))
print("Current edits:", edited())
```
