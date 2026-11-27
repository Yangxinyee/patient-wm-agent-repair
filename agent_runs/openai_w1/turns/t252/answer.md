The records show negative adjusted insulin associations and positive norepinephrine associations, but those one-step estimates are not yet rollout targets. The output was truncated before the candidate inventory. I’ll inspect only the recent calculations and validation results; this should reveal whether the assertion was just a shape error or a substantive flaw.

```python
import re, inspect, numpy as np
from pathlib import Path

_g224 = list(globals().items())
def _describe224(n, v):
    if isinstance(v, np.ndarray):
        return f"{n}: ndarray {v.shape} {v.dtype}"
    if hasattr(v, "shape") and type(v).__module__.startswith("torch"):
        return f"{n}: tensor {tuple(v.shape)}"
    if isinstance(v, dict):
        parts = []
        for k, x in list(v.items())[:28]:
            if isinstance(x, np.ndarray):
                parts.append(f"{k}:{x.shape}")
            elif hasattr(x, "shape") and type(x).__module__.startswith("torch"):
                parts.append(f"{k}:{tuple(x.shape)}")
            else:
                parts.append(f"{k}:{type(x).__name__}")
        return f"{n}: dict " + ", ".join(parts)
    if inspect.isfunction(v):
        try:
            return f"{n}: function {inspect.signature(v)}"
        except Exception:
            return f"{n}: function"
    if isinstance(v, (list, tuple)):
        return f"{n}: {type(v).__name__}[{len(v)}] {repr(v[:3])[:100]}"
    if isinstance(v, (str, int, float, bool, np.number)):
        return f"{n}: {repr(v)[:120]}"
    return f"{n}: {type(v).__name__}"

_recent224 = [(n,v) for n,v in _g224
              if not n.startswith("_") and re.search(r"2(?:0[0-9]|1[0-9]|2[0-2])", n)]
_text224 = "\n".join(_describe224(n,v) for n,v in sorted(_recent224))
print("RECENT OBJECTS:\n" + _text224[:5600])

_names224 = [n for n,v in _g224 if not n.startswith("_")
             and re.search(r"relay|marker|jac|finite|sens|carrier|attn|head|cpu|direction", n, re.I)
             and not re.search(r"2(?:0[0-9]|1[0-9]|2[0-2])", n)]
print("\nOTHER RELEVANT NAMES:\n" + ", ".join(sorted(_names224))[:1400])

_notes224 = Path("/work/notes.md").read_text()
_numeric224 = list(re.finditer(r"(?im)^.*(?:5x|5×|numeric amplification).*$", _notes224))
print("\nEARLIER NUMERIC VALIDATION:")
for _m224 in _numeric224[:4]:
    print(_notes224[max(0,_m224.start()-100):min(len(_notes224),_m224.end()+340)])

with open("/work/notes.md", "a") as _f224:
    _f224.write("\n## Run 224\nRun 223 output was truncated before the candidate inventory. Narrow recovery of recent arrays/functions and earlier numeric-edit validation; no weights changed. Early adjusted associations are not being substituted for dynamic rollout targets.\n")
```