Doubling the action encoder increased factual value NLL by only 0.00009 nats, well within the 0.005 limit. That establishes factual safety for this probe, not better dose effects. I’ll recover the architecture and earlier causal estimates before testing rollouts. Larger, correctly directed dose responses would support the edit; unchanged responses would refute it.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLY NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:11000])
print("\n=== RETAINED OBJECTS ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_"):
        continue
    if isinstance(_obj, (pd.DataFrame, np.ndarray, torch.Tensor)):
        print(_name, type(_obj).__name__, _obj.shape)
    elif isinstance(_obj, dict):
        print(_name, "dict", list(_obj.keys())[:12])
    elif isinstance(_obj, (list, tuple)) and len(_obj) <= 110:
        print(_name, type(_obj).__name__, len(_obj), repr(_obj)[:350])
    elif callable(_obj) and getattr(_obj, "__module__", "") == "__main__":
        print(_name, "function")
print("\n=== TUNING SAMPLE ===")
print(pd.read_csv("/data/tuning.csv", nrows=4).to_string(index=False))
print("\n=== ACTION / TRANSITION / VALUE TENSORS ===")
for _name in NAMES:
    if any(_s in _name for _s in ("encode_action", "transition", "value", "decoder", "last_numeric")):
        print(_name, SHAPES[_name])
print("\nCurrent edits:", edited())
with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Restart — Python 30; forward 5\n"
        "- Action encoder norm scaled by 2: tuning factual NLL 1.66215 "
        "versus original 1.66206, delta +0.00009 on 4,499 readings.\n"
        "- Factual probe passes; dose-rollout response remains untested.\n"
        "- Recovering architecture, early causal analyses, and retained objects "
        "before choosing the next paired rollout.\n"
    )
```