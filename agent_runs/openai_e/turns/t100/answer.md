The trial’s norepinephrine responses are imbalanced: doubling shifts MAP only +1.54, while halving shifts it −3.79. Its insulin-halving response may also have been disrupted. I’ll restore the saved insulin-only baseline and recover the earlier diagnostics. A useful replacement must improve both norepinephrine responses while preserving factual fit and insulin behavior.

```python
import re
from pathlib import Path
import numpy as np
import pandas as pd

baseline85 = np.load("/work/baseline77.npz")["delta"]
save_edit("core.b_in.weight", load_original("core.b_in.weight") + baseline85)
print("RESTORED insulin-only baseline; delta norm:", float(np.linalg.norm(baseline85)))

sections85 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
print("\nSELECTED PRIOR NOTES")
for sec in sections85:
    if re.match(r"## (Run (?:5[3-9]|6[0-9])|Forward (?:[4-9]|10))\b", sec):
        print(sec[:560].strip())

print("\nLOCAL FUNCTIONS")
fn85 = sorted(k for k, v in list(globals().items())
              if callable(v) and getattr(v, "__module__", None) == "__main__")
print(fn85)

print("\nRELEVANT ARRAYS / TABLES")
shown85 = 0
for k, v in sorted(list(globals().items())):
    if any(s in k.lower() for s in ["ne", "81", "82", "e_f", "ef", "direction", "jac", "insulin", "fit", "panel", "ids"]):
        if isinstance(v, np.ndarray):
            print(k, v.shape, str(v.dtype))
            shown85 += 1
        elif isinstance(v, pd.DataFrame):
            print(k, "DataFrame", v.shape, list(v.columns)[:10])
            shown85 += 1
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:8])
            shown85 += 1
        if shown85 >= 48:
            break

print("\nFORWARD TABLE SCHEMAS AND AVAILABLE ROLLOUT SUMMARIES")
for k85 in range(1, 13):
    fp85 = Path(f"/work/fwd_{k85}/dose_edit_edited.csv")
    if not fp85.exists():
        continue
    d85 = pd.read_csv(fp85)
    print("fwd", k85, "columns", list(d85.columns))
    if k85 == 12:
        print(d85.head(2).to_string(index=False))
    ec85 = next((c for c in ["edit", "intervention", "dose_edit"] if c in d85), None)
    sc85 = next((c for c in ["shift", "mean_shift"] if c in d85), None)
    if ec85 is not None and sc85 is not None:
        q85 = d85[d85["reached"].astype(bool)] if "reached" in d85 else d85
        print(q85.groupby(ec85)[sc85].mean().to_dict())

d85f = pd.read_csv("/work/fwd_11/factual_original.csv")
print("\nFACTUAL COLUMNS", list(d85f.columns))
print(d85f.head(1).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 85: reject NE trial and restore insulin baseline\n"
            "Forward12 trial responses: insulin x2 -12.512, half +3.326; NE x2 +1.543, half -3.791. "
            "NE responses are imbalanced and factual NLL failed by +.03339. "
            "Restored original b_in plus baseline77 insulin-only delta. "
            "Recovering prior validation and local forward helpers for a more selective NE edit. "
            "Python runs: 85; forwards: 12.\n")
```