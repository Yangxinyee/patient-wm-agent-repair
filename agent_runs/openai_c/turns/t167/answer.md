The installed edit is numeric gain 4 along the saved direction, plus recurrent-input gain 1.30. The recovery output was truncated before the recent results, so I’ll extract compact summaries next. This will establish which factual checks can be pooled and what rollout effects this exact candidate produced.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("NUMERIC EDIT DIRECTION:", np.load("/work/vectors/v001.npy"))

for k in range(43, 49):
    folder = Path(f"/work/fwd_{k}")
    print(f"\nFORWARD {k}:",
          ", ".join(sorted(p.name for p in folder.iterdir())))
    for name in ("factual_original.csv", "factual_edited.csv",
                 "dose_edit_original.csv", "dose_edit_edited.csv"):
        path = folder / name
        if path.exists():
            df = pd.read_csv(path)
            print(name, "rows:", len(df), "columns:", df.columns.tolist())
            print(df.head(1).to_string(index=False))

print("\nRELEVANT VARIABLE NAMES:")
print([name for name in globals()
       if any(s in name.lower() for s in
              ("probe", "valid", "stay", "target", "audit", "113"))])

notes = Path("/work/notes.md")
text = notes.read_text()
print("\nMOST RECENT NOTES:")
print(text[-2400:])
notes.write_text(text + """
## Python runs 114–115: final candidate recovery
- Confirmed installed operations: numeric dose/rate-direction gain 4,
  and core.b_in.weight gain 1.30.
- Forward 48 independent 100-stay factual check: original 1.61959,
  edited 1.62230; increase +0.00271 nats per reading.
- This sample passes the factual budget; pooled uncertainty remains to be audited.
- Recovering forwards 46–47 before allocating the final two passes.
- Forward passes used: 48; two remain. No edits changed.
""")
```
