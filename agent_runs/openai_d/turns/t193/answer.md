The kernel retains the paired-GELU and local-NLL evaluators, so another NE trial can be reversible. I’ll retrieve the causal evidence and pair formula next. A useful correction must follow the record-supported effect and pass actual factual and rollout checks—not just a synthetic probe.

```python
from pathlib import Path
import inspect, re

_s168 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
_out168 = ["=== IDENTIFICATION NOTES ==="]
for _s in _s168:
    if re.match(r"## Run (?:4|5|6|7|8|9)\b", _s):
        _out168.append(_s[:1400])
print("\n".join(_out168)[:6500])

print("\n=== PAIR / NLL FUNCTION DEFINITIONS ===")
for _n in ["r138_pair", "r138_nll", "r145_hidden", "r145_head", "r159_pair"]:
    _f = globals().get(_n)
    if _f is None:
        continue
    try:
        print(_n, inspect.getsource(_f)[:1200])
    except Exception:
        print(_n, "source unavailable; variables:", _f.__code__.co_varnames,
              "globals:", _f.__code__.co_names)

print("\n=== KEY NE VARIABLES ===")
for _n in ["r138_slots", "r138_tau", "r139_ridge", "r139_scale",
           "r139_target", "r139_error", "r139_loss", "r144_formula",
           "r165_ne_slots", "r165_ne_dv", "r165_ne_gv", "r138_features",
           "r138_y", "r137_fit", "r137_val"]:
    _o = globals().get(_n)
    if hasattr(_o, "shape"):
        print(_n, type(_o).__name__, _o.shape)
    else:
        print(_n, repr(_o)[:450])
```
