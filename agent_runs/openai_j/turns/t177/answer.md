The GPU check shows a value-NLL increase of 0.00307 nats, within the 0.005 limit but with limited headroom. The candidate still needs rollout validation. I’ll recover the record-based effect estimates and stay IDs first; a useful repair must improve the insulin response without materially disturbing MAP or factual fit.

```python
from pathlib import Path
import json, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("\n=== NOTEBOOK HEADINGS ===")
print("\n".join(x for x in _notes.splitlines() if x.startswith("#")))
print("\n=== EARLY CAUSAL EVIDENCE ===")
print(_notes[:22000])

print("\n=== PERSISTENT KERNEL INVENTORY ===")
for _k, _v in list(globals().items()):
    if _k.startswith("_"):
        continue
    if isinstance(_v, (np.ndarray, pd.DataFrame)):
        print(_k, type(_v).__name__, _v.shape)
    elif hasattr(_v, "shape") and type(_v).__module__.startswith("torch"):
        print(_k, type(_v).__name__, tuple(_v.shape))
    elif isinstance(_v, dict):
        print(_k, "dict", len(_v), list(_v.keys())[:12])
    elif isinstance(_v, (list, tuple)) and len(_v) < 120:
        print(_k, type(_v).__name__, len(_v), str(_v)[:350])
    elif callable(_v) and getattr(_v, "__module__", "") == "__main__":
        print(_k, "function")

_f15 = pd.read_csv("/work/fwd_15/factual_original.csv")
print("\n=== FACTUAL FILE SCHEMA AND STAYS ===")
print(_f15.head().to_string(index=False))
print("columns:", _f15.columns.tolist())
for _c in _f15.columns:
    if "stay" in _c.lower():
        print(_c, _f15[_c].drop_duplicates().tolist())

print("\n=== PRIOR ROLLOUT FILES ===")
for _p in sorted(Path("/work").glob("fwd_*/dose_edit_*.csv")):
    _df = pd.read_csv(_p)
    print(str(_p), _df.shape, _df.columns.tolist())
    print(_df.head(2).to_string(index=False))

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Restart, run 160: GPU factual check passed\n"
             "- Forward 15: original value NLL 1.63164; current glucose-only edit 1.63471, delta +0.00307.\n"
             "- MAP factual predictions unchanged. Margin to the +0.005 limit is 0.00193 nats.\n"
             "- Recovering causal evidence, state inventory, and stay IDs before rollout validation.\n")
```