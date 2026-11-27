The stricter gate passed factual fit but nearly silenced the insulin response: doubling gave −2.364 and halving +0.479. That argues against this candidate; norepinephrine remains asymmetric too. I’ll recover the edit construction and effect-estimation units before choosing a less restrictive gate. A successful adjustment should restore the doubled-dose response without reviving the excessive half-dose response.

```python
from pathlib import Path
import types

with open("/work/notes.md", "a") as f:
    f.write("\n## GPU passes 23–24, after Run 246\n"
            "Saved gain-1.5/gate-40 candidate passed factual validation on 98 tuning stays: "
            "original 1.59352, edited 1.59705, delta +.00353. "
            "Reached-row dose shifts were insulin x2 -2.364, half +.479, "
            "NE x2 +.879, half -3.240. The strict gate largely suppresses insulin "
            "transmission, rather than resolving its calibration. Recovering construction "
            "and identification units before selecting an intermediate gate. No new edits.\n")

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

_snapshot247 = list(globals().items())
print("=== RETAINED FUNCTION INTERFACES ===")
for _name247, _obj247 in _snapshot247:
    if isinstance(_obj247, types.FunctionType) and not _name247.startswith("_"):
        _code247 = _obj247.__code__
        print(_name247, _code247.co_varnames[:_code247.co_argcount],
              "defaults=", repr(_obj247.__defaults__)[:160])

print("=== RELEVANT RETAINED VARIABLES ===")
_tokens247 = ("r244", "r245", "r246", "backup", "kernel", "target",
              "gain", "gate", "innovation", "policy", "base", "carrier")
for _name247, _obj247 in _snapshot247:
    if not _name247.startswith("_") and any(t in _name247.lower() for t in _tokens247):
        if isinstance(_obj247, types.FunctionType):
            continue
        if isinstance(_obj247, dict):
            _desc247 = "keys=" + repr(list(_obj247.keys())[:30])
        elif hasattr(_obj247, "shape"):
            _desc247 = "shape=" + repr(tuple(_obj247.shape))
        elif isinstance(_obj247, (int, float, str, tuple, list)):
            _desc247 = repr(_obj247)[:220]
        else:
            _desc247 = type(_obj247).__name__
        print(_name247, type(_obj247).__name__, _desc247)
```
