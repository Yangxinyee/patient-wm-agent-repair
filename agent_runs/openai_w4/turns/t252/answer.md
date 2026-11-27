The dose channels transmit half and doubled doses comparably, so transmission alone does not explain the rollout asymmetry. I’ll inspect the recurrent dynamics and hourly responses. A response that changes sign or grows disproportionately over time would point to feedback rather than dose encoding.

```python
import os, json, inspect, numpy as np, pandas as pd
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RETAINED IMPULSE OBJECTS ===")
for name in ["impulse_tables_repair", "impulse_results159", "impulse_x159",
             "impulse_q159", "impulse_p159", "impulse_mean159",
             "impulse_time159", "impulse_hour160", "impulse_tests166"]:
    obj = globals().get(name)
    print(name, type(obj).__name__)
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:12]:
            print(" ", k, type(v).__name__, getattr(v, "shape", None),
                  list(v)[:8] if isinstance(v, dict) else "")
    elif hasattr(obj, "shape"):
        print(" ", obj.shape)
        if isinstance(obj, pd.DataFrame):
            print(obj.head(3).to_string(index=False))
    elif isinstance(obj, (list, tuple)):
        print(" length", len(obj), "first", repr(obj[:2])[:1200])

print("\n=== RESPONSE/CHANNEL FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and any(s in name.lower()
            for s in ("impulse", "channel", "dual", "direction", "transition73", "make_ne")):
        try:
            print(name, inspect.signature(obj), "globals:", obj.__code__.co_names)
        except Exception:
            pass

print("\n=== CURRENT ROLLOUT: ELIGIBLE-HOUR ROW MEANS ===")
for model in ["original", "edited"]:
    d219 = pd.read_csv(f"/work/fwd_21/dose_edit_{model}.csv")
    eligible219 = d219[d219.reached.astype(str).str.lower().eq("true")]
    print(model)
    print(eligible219.groupby(["hour", "edit"])["shift"].mean()
          .unstack().round(3).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 219\nRecovered architecture, retained impulse interfaces and latest hourly rollout profiles. "
            "Fixed-context dose transmission did not explain half/double asymmetry; investigating recurrent/generated-reading "
            "feedback before another weight edit. No saved weights changed.\n")
```
