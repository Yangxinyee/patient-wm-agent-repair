The records include threshold-based and history-adjusted analyses, and earlier conflicting estimates were explicitly rejected. I’ll recover the accepted response kernels and the broader-anchor fit grid. That will show whether stronger candidates were already ruled out by factual loss or remain worth testing.

```python
import dis
import numpy as np
from pathlib import Path

def compact240(v, depth=0):
    if hasattr(v, "shape"):
        return f"{type(v).__name__}{tuple(v.shape)}"
    if isinstance(v, dict):
        if depth >= 1:
            return "dict keys=" + repr(list(v.keys())[:12])
        return {str(k): compact240(x, depth + 1) for k, x in list(v.items())[:16]}
    if isinstance(v, (list, tuple)):
        return [compact240(x, depth + 1) for x in v[:12]]
    return repr(v)[:180]

for n in ["key235", "expanded234", "stats234", "paired214", "headbase214",
          "pruneorder214", "family204", "gatecanon200", "kernels19",
          "directions207", "funcs230", "s231"]:
    if n in globals():
        print(n, compact240(globals()[n]))
print("\nTOP FIT GRID RESULTS:")
if isinstance(stats234, list):
    for v in stats234[:8]:
        print(repr(v)[:650])
elif hasattr(stats234, "head"):
    print(stats234.head(12).to_string())

print("\nMAKE_ARCH215:")
dis.dis(make_arch215)
print("\nACCEPTED RECORD EVIDENCE:")
print(Path("/work/notes.md").read_text()[5500:10800])
```