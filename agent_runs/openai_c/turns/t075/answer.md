The records support an immediate glucose decrease of about 2 units per insulin-rate unit and a MAP increase of about 3 per norepinephrine-rate unit. These are estimates with assumptions, not observed counterfactual truth. I’ll check which gains were already tested and inspect drug-specific embedding paths; useful amplification should preserve factual fit and move the rollouts toward those estimates.

```python
from pathlib import Path
import numpy as np

notes = Path("/work/notes.md").read_text()
print("NOTEBOOK CONTINUATION\n", notes[6200:12000])

print("\nRELEVANT TENSORS")
for name in NAMES:
    if any(s in name for s in [
        "encode_action", "embed.bins", "last_numeric",
        "embed_transition", "attr_heads.value", "query_norm"
    ]):
        print(name, SHAPES[name])

print("\nFORWARD FILES")
for k in [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
    p = Path(f"/work/fwd_{k}")
    print(k, [q.name for q in p.iterdir()] if p.exists() else "missing")

print("\nHEAD SURROGATE FRAME COLUMNS")
if "head_surrogates" in globals():
    for event, item in head_surrogates.items():
        print(event, "keys", list(item),
              "frame columns", list(item["frame"].columns) if "frame" in item else None)

print("\nGAIN-4 DIRECTION")
candidate = repair_candidates["numeric_gain4"][0]
print({k: (np.asarray(v).tolist() if k == "vector" and not isinstance(v, str) else v)
       for k, v in candidate.items()})

print("\nSMALL REMAINING VARIABLES")
for name in sorted(list(globals())):
    if name.startswith(("target", "persist", "record", "roll", "validation", "dose_", "norm")):
        obj = globals()[name]
        if callable(obj):
            print(name, "callable")
        elif isinstance(obj, dict):
            print(name, "keys", list(obj)[:15])
        elif hasattr(obj, "shape"):
            print(name, "shape", obj.shape)
        elif isinstance(obj, (list, tuple)):
            print(name, "length", len(obj), "first", obj[:5])
        elif isinstance(obj, (str, int, float)):
            print(name, obj)
```
