The architecture confirms that the edit changes only the value head, though generated readings can still affect later states. The inspection output was truncated before the key estimates and stay IDs. I’ll extract those compactly so the rollout test can distinguish a genuine improvement from feedback drift.

```python
from pathlib import Path
import re, pandas as pd, numpy as np

_notes = Path("/work/notes.md").read_text()
for _title in [
    "Baseline rollout and flexible residualisation",
    "Causal identification checks",
    "Dose-innovation local projections",
    "Sensor-proxy dynamic check",
    "Fixed-dose response-kernel estimates",
]:
    _match = re.search(r"(?m)^## " + re.escape(_title) + r"\n(.*?)(?=^## |\Z)",
                       _notes, flags=re.S | re.M)
    if _match:
        print("\n##", _title)
        print(_match.group(1)[:1500])

_f15 = pd.read_csv("/work/fwd_15/factual_original.csv")
print("\nFACTUAL COLUMNS:", _f15.columns.tolist())
print("FACTUAL FIRST ROW:", _f15.iloc[0].to_dict())
_staycol = next(c for c in _f15.columns if "stay" in c.lower())
rollout_ids = _f15[_staycol].drop_duplicates().tolist()
print("ROLLOUT IDS:", rollout_ids)

print("\nFUNCTIONS:")
print([k for k, v in list(globals().items())
       if not k.startswith("_") and callable(v)
       and getattr(v, "__module__", "") == "__main__"])
print("\nRELEVANT ARRAYS/DICTS:")
for _k, _v in list(globals().items()):
    if _k.startswith("_") or not re.search(
        r"candidate|gate|raw|stoch|anchor|dose|kernel|target|meta|panel|gain|coef|ridge", _k, re.I
    ):
        continue
    if isinstance(_v, (np.ndarray, pd.DataFrame)):
        print(_k, type(_v).__name__, _v.shape)
    elif isinstance(_v, dict):
        print(_k, "dict", list(_v)[:12])
    elif isinstance(_v, (float, int, str)):
        print(_k, str(_v)[:100])

print("\nROLLOUT FILE SCHEMAS:")
for _p in sorted(Path("/work").glob("fwd_*/dose_edit_original.csv"))[:2]:
    _d = pd.read_csv(_p)
    print(_p, _d.shape, _d.columns.tolist())
    print(_d.head(2).to_string(index=False))
```