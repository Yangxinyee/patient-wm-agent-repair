The early notebook shows substantial treatment-selection bias: adjusting for pre-dose history changes the apparent drug effects. It also records small insulin responses and negative norepinephrine responses from the original model. I’ll recover the later robustness checks and current calibration variables; consistent estimates across those checks would justify the repair targets.

```python
import re
from pathlib import Path

_repair_notes = Path("/work/notes.md").read_text()
_repair_start = _repair_notes.find("## Run 4\n")
print("=== CAUSAL ROBUSTNESS AND INITIAL REPAIR ===")
print(_repair_notes[_repair_start:_repair_start + 5200])

_repair_vars = ", ".join(sorted(
    k for k in list(globals())
    if not k.startswith("_") and k not in {"In", "Out", "get_ipython", "exit", "quit"}
))
print("\n=== KERNEL VARIABLE NAMES ===")
print(_repair_vars[:3400])

print("\n=== FORWARD 39 CSV LAYOUT ===")
import pandas as pd
for _repair_model in ("original", "edited"):
    _repair_df = pd.read_csv(
        f"/work/fwd_39/factual_{_repair_model}.csv"
    )
    print(_repair_model, _repair_df.shape, _repair_df.columns.tolist())
    print(_repair_df.head(1).to_string(index=False))
```