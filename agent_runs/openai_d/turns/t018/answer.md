The reconstruction errors are larger than the estimated sampling noise, so the reading head is still missing something. I’ll recheck the architecture and saved results before editing weights. An exact feature layout should reduce the error to sampling noise; if it does not, I’ll avoid relying on this reconstruction.

```python
from pathlib import Path
import json, numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

print("\n=== PERSISTENT VARIABLES ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), "keys:", list(v)[:12])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v), repr(v[:4])[:300])
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function")

print("\n=== FORWARD FILES ===")
for d in sorted(Path("/work").glob("fwd_*")):
    print(d.name, [p.name for p in d.iterdir()])
    for p in sorted(d.glob("dose_edit_*.csv")):
        q = pd.read_csv(p)
        print(p.name, q.shape, q.columns.tolist())
        print(q.head(8).to_string(index=False))

print("\n=== NOTEBOOK BEGINNING ===")
print(Path("/work/notes.md").read_text()[:12500])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart / Run 14\n"
            "No edits. Head reconstruction error exceeds four-draw Monte Carlo noise; "
            "recovering architecture, persistent variables, and baseline rollout results "
            "before attempting a causal weight edit.\n")
```