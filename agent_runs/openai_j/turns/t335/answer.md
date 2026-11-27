The latest rollout strengthens insulin’s response substantially, while norepinephrine’s response remains smaller. These are predictions, not proof of the true effects. I’ll recover the records-based estimates and this candidate’s factual-loss audit; independent factual validation will determine whether it is safe to retain.

```python
import os, re, json, numpy as np, pandas as pd
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes286 = Path("/work/notes.md").read_text()
print("\n=== LATEST NOTES ===")
print(notes286[-5200:])

print("\n=== RECORD-BASED ESTIMATES ===")
for fn286 in ["kernel_targets.csv", "dml_results.csv", "dynamic_iv.csv", "sensor_iv.csv"]:
    p286 = Path("/work") / fn286
    if p286.exists():
        d286 = pd.read_csv(p286)
        print(fn286, "shape", d286.shape)
        print(d286.head(12).to_string(index=False))

print("\n=== CURRENT EDIT / CANDIDATE ARCHIVES ===")
print("edited:", edited())
for fn286 in ["calibrated_backup285.npz", "safety_candidate285.npz"]:
    with np.load("/work/" + fn286, allow_pickle=False) as a286:
        print(fn286, [(k, a286[k].shape) for k in a286.files])

print("\n=== REUSABLE FUNCTIONS ===")
print(sorted(k for k, v in list(globals().items())
             if callable(v) and getattr(v, "__module__", "") in ["__main__", None]))
print("\n=== RECENT VARIABLE NAMES ===")
print(sorted(k for k in list(globals())
             if re.search(r"(26[3-9]|28[0-5]|safety|calibr|frontier|candidate|builder|factual|tuning)", k))[:220])

print("\n=== RECENT FORWARD FILES ===")
for n286 in [38, 39, 40, 41, 42]:
    p286 = Path(f"/work/fwd_{n286}")
    if p286.exists():
        print(str(p286), sorted(x.name for x in p286.iterdir()))

with open("/work/notes.md", "a") as f286:
    f286.write("\n## Run 286: restart recovery\n"
               "- Forward 40 current safety candidate: insulin -10.008/+5.254; NE +0.940/-1.505 on 100 tuning stays.\n"
               "- These are model rollout predictions, not intervention truth.\n"
               "- Recovering records-derived estimates and factual safety evidence; no weights changed.\n"
               "- Counts: 286 Python runs, 40 forward passes.\n")
```