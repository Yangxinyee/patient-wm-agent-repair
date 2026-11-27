The architecture confirms that drug effects must reach the value head through the model’s state. The recovery output was truncated before the stay IDs. I’ll retrieve those IDs and the saved effect estimates so the next paired rollout uses the same factual-audit sample.

```python
import json, inspect, copy, pandas as pd
from pathlib import Path

audit188 = pd.read_csv("/work/fwd_12/factual_original.csv")
print("FACTUAL COLUMNS:", audit188.columns.tolist())
idcol188 = next(c for c in audit188.columns if "stay" in c.lower())
ids188 = audit188[idcol188].drop_duplicates().astype(int).tolist()
print("NEXT FORWARD REQUEST:")
print(json.dumps({"type": "dose_edit", "model": "both",
                  "split": "tuning", "stays": ids188}))

snapshot188 = {name: copy.deepcopy(load(name)) for name in edited()}
print("EDIT TYPES:", {name: type(value).__name__
                      for name, value in snapshot188.items()})

for name in ["kernels19", "readout179", "targets179", "target185",
             "rows173", "rows179", "cf_features", "cf_features178",
             "cf_features180"]:
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, dict):
            print(name, "keys:", list(obj)[:20])
            if name == "kernels19":
                print(obj)
        elif hasattr(obj, "shape"):
            print(name, "shape:", tuple(obj.shape))
            if isinstance(obj, pd.DataFrame):
                print("columns:", obj.columns.tolist())
        else:
            print(name, str(obj)[:500])

print("RELEVANT FUNCTIONS:")
for name, obj in sorted(list(globals().items())):
    if callable(obj) and (
        re.search(r"(decoder|features|collect|step|probe|bundle)", name, re.I)
        or name in ["f176", "f177"]
    ):
        try:
            print(name, inspect.signature(obj))
        except (ValueError, TypeError):
            pass

print("LATEST NOTE BEFORE RESTART:")
notes188 = Path("/work/notes.md").read_text()
pos188 = notes188.find("## Runs 179")
print(notes188[pos188:pos188 + 1800] if pos188 >= 0 else notes188[-1800:])
```