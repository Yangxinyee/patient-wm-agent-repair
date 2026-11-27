The current candidate’s factual-loss increase is +0.00289 nats on the first 100 tuning stays, below the limit. The records-based estimates support stronger insulin and norepinephrine effects than the original predicts, though those estimates depend on identification assumptions. I’ll identify the exact candidate and validation groups; agreement across the remaining stays would strengthen the safety evidence.

```python
import numpy as np, pandas as pd, json
from pathlib import Path

print("=== COMPLETE RECORD-BASED ROLLOUT TARGETS ===")
print(pd.read_csv("/work/kernel_targets.csv").to_string(index=False))

print("\n=== CURRENT CANDIDATE METADATA ===")
for name288 in ["candidate285", "sort285", "tab285", "selected283"]:
    if name288 in globals():
        obj288 = globals()[name288]
        print(name288, type(obj288).__name__)
        if isinstance(obj288, pd.DataFrame):
            print(obj288.head(4).to_string(index=False))
        elif isinstance(obj288, pd.Series):
            print(obj288.to_string())
        elif isinstance(obj288, (str, int, float, tuple, list)):
            print(str(obj288)[:1200])
        elif isinstance(obj288, dict):
            print({k: (v.shape if hasattr(v, "shape") else str(v)[:100])
                   for k, v in list(obj288.items())[:12]})

ids288 = np.sort(pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique())
print("\n=== TUNING IDS ===")
print("count", len(ids288), "first", ids288[:3], "last", ids288[-3:])
print("contiguous:", bool(np.array_equal(ids288, np.arange(ids288[0], ids288[0]+len(ids288)))))
for group288 in range(4):
    print("group", group288, "range", int(ids288[100*group288]), int(ids288[100*group288+99]))

print("\n=== RECENT FORWARD SUMMARIES ===")
for n288 in [36, 37, 38, 39, 40]:
    p288 = Path(f"/work/fwd_{n288}/summary.json")
    if p288.exists():
        s288 = json.loads(p288.read_text())
        print("fwd", n288, json.dumps(s288)[:1150])

print("\n=== CURRENT ARCHIVE MATCH ===")
with np.load("/work/safety_candidate285.npz", allow_pickle=False) as a288:
    for k288 in a288.files:
        cur288 = load(k288)
        if hasattr(cur288, "detach"):
            cur288 = cur288.detach().cpu().numpy()
        print(k288, "max difference", float(np.max(np.abs(np.asarray(cur288)-a288[k288]))))

with open("/work/notes.md", "a") as f288:
    f288.write("\n## Run 288: final validation plan\n"
               "- Current safety candidate factual dNLL +0.00289406 on first 100 tuning stays (4,751 readings).\n"
               "- Need remaining three factual groups and independent dose-rollout groups before submission.\n"
               "- Records-derived kernel targets remain assumption-dependent; no weights changed.\n"
               "- Counts: 288 Python runs, 40 forward passes.\n")
```