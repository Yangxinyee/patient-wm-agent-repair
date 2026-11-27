The second 100-stay batch shows no factual penalty: NLL improved by 0.00012 nats. This supports keeping the dose-invariant edit, but does not establish that the true effects are zero. I’ll recover the architecture and analysis objects before checking for overlooked nonlinear effects or alternative dose-input paths; either finding would challenge the current edit.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())
print("\n=== RETAINED OBJECTS ===")
for key in sorted(list(globals())):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    obj = globals()[key]
    if isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray, dict, list, tuple)):
        desc = str(getattr(obj, "shape", f"len={len(obj)}"))
        if isinstance(obj, pd.DataFrame):
            desc += " columns=" + repr(list(obj.columns)[:45])
        elif isinstance(obj, dict):
            desc += " keys=" + repr(list(obj.keys())[:25])
        print(key, type(obj).__name__, desc)
    elif callable(obj) and getattr(obj, "__module__", None) == "__main__":
        print(key, "function")
print("\n=== CURRENT EDITS ===")
print(edited())
print("\n=== VOCAB TOP LEVEL ===")
print(list(VOCAB.keys()) if isinstance(VOCAB, dict) else type(VOCAB))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Restart / run 11\n"
        "Forward 5 (fwd_6), second 100 tuning stays: original factual NLL "
        "1.49403, edited 1.49391, difference -0.00012. Together with first "
        "batch, no observed factual penalty. Candidate A still enforces "
        "exact dose-magnitude invariance. Next: recover architecture and "
        "analysis objects, then check nonlinear/high-dose effects and "
        "validate remaining tuning stays before accepting zero effects.\n"
    )
```