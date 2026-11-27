The recovered records support persistent glucose-lowering insulin effects and MAP-raising norepinephrine effects, subject to the history-adjustment assumptions. The output cut off before the current trial’s factual results. I’ll retrieve those narrowly; exceeding the 0.005-nat limit would rule this trial out.

```python
from pathlib import Path
import numpy as np, pandas as pd, inspect, types

print("=== FORWARD 27 FACTUAL FIT ===")
for p in sorted(Path("/work/fwd_27").glob("*.csv")):
    q27 = pd.read_csv(p)
    print(p.name, "rows", len(q27), "columns", list(q27.columns))
    print({
        c: float(pd.to_numeric(q27[c], errors="coerce").mean())
        for c in q27 if "nll" in c.lower() or "loss" in c.lower()
    })
for p in sorted(Path("/work/fwd_27").glob("*.json")):
    print(p.name, p.read_text()[:1800])

print("\n=== ARCHIVE KEYS ===")
for fn in ["map_regularized_trial.npz", "map_feature_trial_backup.npz",
           "glucose_115_candidate.npz"]:
    with np.load("/work/" + fn, allow_pickle=False) as arc:
        print(fn, [(k, arc[k].shape) for k in arc.files])
        for k in arc.files:
            if arc[k].size <= 8:
                print(k, arc[k])

print("\n=== FUNCTION NAMES ===")
resume_functions = {
    n: o for n, o in list(globals().items())
    if isinstance(o, types.FunctionType) and o.__module__ == "__main__"
}
print(" ".join(sorted(resume_functions)))
print("\n=== SELECTED SIGNATURES ===")
for n, o in sorted(resume_functions.items()):
    if any(t in n.lower() for t in ("rec", "route", "predict", "inject", "case", "fisher", "slot")):
        print(n, inspect.signature(o))

print("\n=== SELECTED OBJECTS ===")
for n, o in sorted(list(globals().items())):
    if n.startswith("_") or not any(t in n.lower() for t in
        ("target", "route", "fisher", "rec_", "regular", "trial", "valid", "case", "b0")):
        continue
    if isinstance(o, np.ndarray):
        print(n, o.shape)
    elif isinstance(o, dict):
        print(n, "dict", len(o), "keys", list(o)[:6])
    elif isinstance(o, pd.DataFrame):
        print(n, "DataFrame", o.shape)
    elif isinstance(o, (str, int, float, bool, tuple, list)) and len(repr(o)) < 180:
        print(n, repr(o))

print("\n=== RECORD-TARGET NOTE EXCERPTS ===")
target_lines = [
    s for s in Path("/work/notes.md").read_text().splitlines()
    if any(t in s.lower() for t in
           ("target shift", "target mean", "rollout target", "true shift",
            "convolution", "record-based target", "estimated cumulative"))
]
for s in target_lines[-12:]:
    print(s[:350])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 283: narrow eligibility and interface recovery\n"
        "- Read architecture and early record identification evidence in run 282; output truncated before current factual fit.\n"
        "- Retrieving forward 27 NLL, candidate keys and compact interfaces without repeating architecture.\n"
        "- No weights changed. Python runs: 283; forward passes: 28.\n"
    )
```