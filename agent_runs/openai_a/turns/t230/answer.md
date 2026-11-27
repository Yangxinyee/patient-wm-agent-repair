The architecture confirms that the edit changes reading predictions, not dose encoding. The inspection output was truncated, so I’ll retrieve a compact index and the tuning IDs. Then a paired rollout can test whether the stronger edit improves the insulin effect.

```python
import pandas as pd
import re

_restart_tuning_ids = (
    pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"]
    .drop_duplicates().to_numpy()
)
print("FIRST 100 TUNING IDS:")
print(_restart_tuning_ids[:100].tolist())

_items = list(globals().items())
print("\nCUSTOM FUNCTIONS:")
print([n for n, v in _items
       if not n.startswith("_") and callable(v)
       and getattr(v, "__module__", "") == "__main__"])

print("\nPUBLIC VARIABLE NAMES:")
print(" ".join(n for n, v in _items
               if not n.startswith("_") and not callable(v)))

print("\nCHECKPOINT/CONTRAST OBJECTS:")
for n, v in _items:
    if n.startswith("_") or not re.search(
        r"checkpoint|r3|separator|selective|contrast|finger_scale|sc_", n
    ):
        continue
    if isinstance(v, dict):
        print(n, "dict", len(v), "keys", list(v)[:12])
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        print(n, tuple(v.shape))
    elif isinstance(v, (int, float, str, bool)):
        print(n, repr(v)[:100])
    elif callable(v):
        print(n, "function")

print("\nEARLY RECORD-BASED FINDINGS:")
_notes = Path("/work/notes.md").read_text()
print(_notes[:6500])
```
