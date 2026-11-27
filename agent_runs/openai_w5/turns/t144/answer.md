The recovered notebook suggests glucose-lowering insulin effects and MAP-raising norepinephrine effects, while the original rollouts respond only weakly. Those estimates still depend on observational assumptions. The output was truncated before the current helpers appeared, so I’ll retrieve a compact inventory and tuning IDs. Then paired GPU tests can establish whether the new channel preserves factual fit and transports doses correctly.

```python
import types
import inspect
from pathlib import Path

print("=== EARLY EFFECT VALIDATION NOTES ===")
print(Path("/work/notes.md").read_text()[4300:8200])

_snapshot = list(globals().items())
_funcs = []
_objects = []
for _n, _v in _snapshot:
    if _n.startswith("_") or isinstance(_v, types.ModuleType):
        continue
    if inspect.isfunction(_v):
        try:
            _sig = str(inspect.signature(_v))
        except Exception:
            _sig = ""
        _funcs.append(_n + _sig)
    elif _n not in ("NAMES", "SHAPES", "VOCAB"):
        _objects.append(_n + (str(tuple(_v.shape)) if hasattr(_v, "shape") else ""))
print("=== FUNCTIONS ===")
print("\n".join(_funcs))
print("=== OBJECT NAMES / SHAPES ===")
print(", ".join(_objects))

_tuning_ids = pd.read_csv("/data/tuning.csv", usecols=["stay_id"]).stay_id.unique()
print("=== FIRST 100 TUNING IDS ===")
print(_tuning_ids[:100].tolist())
print("=== HISTORY AVAILABILITY ===")
for _n in ("In", "_ih", "history", "code_history"):
    if _n in globals():
        print(_n, type(globals()[_n]).__name__, len(globals()[_n]))
```