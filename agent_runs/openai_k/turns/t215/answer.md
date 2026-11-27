The earlier zero-correction candidate reproduced the original rollout, so its baseline compression appears sound. The failure is in the added drug-response directions. I’ll inspect the surrogate fitted to official predictions and the reserved dose channels; if that surrogate accurately predicts held-out official changes, it can guide a new direction without relying on the rejected head reconstruction.

```python
import numpy as np, torch, re
from pathlib import Path

print("=== OFFICIAL READOUT SURROGATE ===")
for k in ("diagnostics144", "models144", "W144", "candidates90", "post66_base_weights"):
    if k not in globals():
        print(k, "MISSING")
        continue
    v = globals()[k]
    if isinstance(v, dict):
        print(k, "keys", list(v)[:12])
        if k == "diagnostics144":
            print(str(v)[:1800])
        elif k == "models144":
            for a, b in v.items():
                if isinstance(b, dict):
                    print(a, {c: (type(d).__name__, getattr(d, "shape", None)) for c, d in b.items()})
                else:
                    print(a, type(b).__name__, str(b)[:250])
    else:
        print(k, type(v).__name__, getattr(v, "shape", None))

print("\n=== SURROGATE AND CHANNEL HELPERS ===")
for k, v in list(globals().items()):
    if callable(v) and hasattr(v, "__code__") and (
        re.search(r"144|151|154|89|66", k)
        or k in ("r49_readouts", "repaired_core_local", "r59_core", "r59_features")):
        c = v.__code__
        print(k, "args", c.co_varnames[:c.co_argcount],
              "uses", c.co_names[:35], "locals", c.co_varnames[c.co_argcount:c.co_argcount+12])

print("\n=== SMALL CHANNEL AND CALIBRATION ARRAYS ===")
for k in ("Bcomp89", "Binnov89", "V88", "D89", "token_mu88",
          "fi71", "rr33_stays", "rr33_hours", "rr33_u_all",
          "future_index59", "observed59", "r49_register", "dt_next151"):
    if k in globals():
        v = globals()[k]
        print(k, type(v).__name__, getattr(v, "shape", None))
        if hasattr(v, "shape") and np.prod(v.shape) <= 50:
            print(v)

print("\n=== SURROGATE VALIDATION NOTE ===")
for s in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    if re.match(r"## Run[s]? (?:14[4-9]|15[0-4])\b", s):
        print(s[:550])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 197: inspect authoritative surrogate\n"
            "- Zero-channel compressed baseline matched original official rollout. Drug directions, rather than baseline compression, need replacement.\n"
            "- Inspecting retained official-prediction surrogate and physical-dose-channel interfaces.\n"
            "- Python runs: 197; forward passes: 12. Saved edits unchanged.\n")
```
