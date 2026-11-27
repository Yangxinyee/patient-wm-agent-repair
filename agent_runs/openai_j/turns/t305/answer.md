The architecture confirms that the current edit changes only the value head. The diagnostic output was truncated before the candidate builders appeared, so I’ll recover those with a bounded printout. Then I can test a calibrated candidate; lower cached loss alone would not confirm better treatment responses.

```python
import re, inspect
import numpy as np
import pandas as pd
from pathlib import Path

_parts267 = []
_fn267 = sorted(k for k, v in list(globals().items())
                if inspect.isfunction(v) and not k.startswith("_"))
_parts267.append("FUNCTION NAMES:\n" + ", ".join(_fn267))

_parts267.append("\nRECENT VARIABLES:")
_recent267 = [(k, v) for k, v in list(globals().items())
              if not k.startswith("_") and
              any(s in k for s in ("265", "264", "263", "243", "247", "242"))]
_recent267.sort(key=lambda kv: (
    0 if "265" in kv[0] else 1 if "264" in kv[0] else
    2 if "263" in kv[0] else 3, kv[0]))
for k, v in _recent267[:55]:
    if isinstance(v, np.ndarray):
        desc = f"array {v.shape} {v.dtype}"
    elif isinstance(v, pd.DataFrame):
        desc = f"DataFrame {v.shape}; columns={list(v.columns)}"
    elif isinstance(v, dict):
        desc = f"dict {len(v)}; keys={list(v.keys())[:12]}"
    elif inspect.isfunction(v):
        desc = "function parameters=" + str(list(inspect.signature(v).parameters))
    else:
        desc = repr(v)
    _parts267.append(k + ": " + desc[:240])

_parts267.append("\nSAVED WEIGHTS:")
for fn in ("safer_candidate247.npz", "soft_hidden_betas264.npz",
           "best_combined_v1.npz", "validated_gain8_backup243.npz"):
    with np.load("/work/" + fn) as z:
        _parts267.append(fn + ": " +
                         str([(k, z[k].shape) for k in z.files[:12]]))

_cf267 = pd.read_csv("/work/calibration_strength_frontier265.csv")
_parts267.append("\nFRONTIER COLUMNS:\n" + str(list(_cf267.columns)))
_cols267 = [c for c in _cf267.columns
            if any(s in c.lower() for s in
                   ("key", "gain", "strength", "tune", "cv_dnll",
                    "response", "double", "half", "moment"))]
_parts267.append(_cf267[_cols267[:12]].head(8).to_string(index=False))

_out267 = "\n".join(_parts267)
print(_out267[:8500])
print("\nDiagnostic characters:", len(_out267), "; current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 267: bounded candidate recovery\n"
            "- Architecture confirms the current edit is confined to the value head.\n"
            "- Recovering recent candidate variables and builders with bounded output.\n"
            "- No weights changed; cached factual improvements still require rollout validation.\n")
```