The architecture confirms that doses affect the recurrent state, while the current edit changes only the reading head. I’ll recover the saved construction and validation helpers before testing the stronger edit. If its rollout response scales as intended, that supports the head correction; saturation or worse factual fit would argue against further amplification.

```python
import re
import numpy as np
import pandas as pd

g114 = dict(globals())
print("=== NOTEBOOK HELPERS ===")
print(sorted(k for k, v in g114.items()
             if callable(v) and getattr(v, "__module__", "") == "__main__"))
print("\n=== RECENT VARIABLES ===")
for k in sorted(g114):
    if re.search(r"^(r(99|10[0-9]|11[0-2])_|.*(tuning_ids|train_ids|tune_ids|candidate|amp_eval|install).*)$", k):
        v = g114[k]
        shape = getattr(v, "shape", None)
        if isinstance(v, dict):
            desc = "keys=" + repr(list(v.keys())[:25])
        elif isinstance(v, (list, tuple)):
            desc = f"len={len(v)}; first={repr(v[:5])[:200]}"
        else:
            desc = repr(v)[:140] if shape is None and not callable(v) else ""
        print(k, type(v).__name__, shape, desc)
print("\n=== CAUSAL-ESTIMATION NOTES ===")
p114 = notes_text.find("## Run 4")
print(notes_text[p114:p114 + 4500])
print("\n=== FORWARD 12 STAYS ===")
f114 = pd.read_csv("/work/fwd_12/factual_original.csv")
print("columns", f114.columns.tolist())
print("stay range", f114.stay_id.min(), f114.stay_id.max(),
      "count", f114.stay_id.nunique())
```
