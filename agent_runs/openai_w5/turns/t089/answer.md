The architecture confirms that doses affect recurrent state, while reading predictions also use the last-reading registers. A useful edit must therefore change dose sensitivity without disturbing factual state tracking. I’ll inspect the corrected recurrence, existing causal estimates, and original rollout results; this should show which candidate can be tested rather than merely fitted locally.

```python
import dis, json, numpy as np, pandas as pd
from pathlib import Path

print("=== NOTEBOOK BEGINNING ===")
print(Path("/work/notes.md").read_text()[:12500])

print("\n=== USER FUNCTIONS ===")
for k, v in sorted(list(globals().items())):
    if callable(v) and getattr(v, "__module__", None) == "__main__" and hasattr(v, "__code__"):
        print(k, inspect.signature(v), "calls:", v.__code__.co_names)

print("\n=== DATA / CANDIDATE OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or callable(v):
        continue
    if isinstance(v, pd.DataFrame):
        print(k, "DataFrame", v.shape, list(v.columns)[:22])
    elif isinstance(v, dict) and any(s in k.lower() for s in
            ("cache", "head", "spec", "fit", "state", "target", "data")):
        print(k, "dict", {str(a): (tuple(b.shape) if hasattr(b, "shape")
                                  and not callable(b.shape) else type(b).__name__)
                          for a, b in list(v.items())[:22]})
    elif isinstance(v, np.ndarray) and any(s in k.lower() for s in
            ("direction", "response", "coef", "target", "basis", "delta", "candidate")):
        print(k, v.shape, "range", np.nanmin(v), np.nanmax(v))

print("\n=== ORIGINAL FORWARD SUMMARIES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    for file in sorted(folder.glob("*.json")):
        print(str(file), file.read_text()[:2000])
    for file in sorted(folder.glob("dose_edit_original.csv")):
        x = pd.read_csv(file)
        print(str(file), x.shape, x.head(8).to_string(index=False))

for name in ("response_pair68", "latent_step66", "causal_features"):
    print("\n=== IMPLEMENTATION:", name, "===")
    dis.dis(globals()[name])
```