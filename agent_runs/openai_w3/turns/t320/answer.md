The architecture confirms that sampled readings feed back through both recurrent state and each sensor’s “last value” register. Either could amplify a small local error during rollout. The previous output was truncated, so I’ll recover only the essential helpers and estimates. This should show whether we can test accumulation directly without another speculative edit.

```python
import re, inspect, numpy as np, torch
from pathlib import Path

g282 = dict(globals())
print("FUNCTION NAMES")
fn282 = [(n, v) for n, v in sorted(g282.items())
         if callable(v) and hasattr(v, "__code__") and not n.startswith("_")]
print(" ".join(n for n, v in fn282))

print("\nLATEST OBJECTS / IMPORTANT CACHES")
for n, v in sorted(g282.items()):
    m = re.search(r"(\d+)$", n)
    if not (m and int(m.group(1)) >= 278 or
            re.search(r"checkpoint273|checkpoint267|checkpoint222|direction|effect|rho|coef|calib|rollout|factual|carrier|causal", n, re.I)):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        print(n, type(v).__name__, tuple(v.shape),
              repr(np.asarray(v.cpu() if isinstance(v, torch.Tensor) else v).flatten()[:5])[:110])
    elif isinstance(v, dict):
        print(n, "dict", len(v), "keys", repr(list(v)[:15])[:260])
    elif callable(v):
        try: print(n, inspect.signature(v))
        except Exception: pass
    elif isinstance(v, (list, tuple)):
        print(n, type(v).__name__, len(v), repr(v[:3])[:180])
    elif isinstance(v, (int, float, str, np.number)):
        print(n, repr(v)[:180])

print("\nTRANSITION AND TOKEN HELPER INTERFACES")
for n, v in fn282:
    if re.search(r"prelude155|recur|coda|step|tokens|posterior|calib|replay", n, re.I):
        print(n, inspect.signature(v), "names", v.__code__.co_names,
              "constants", repr(v.__code__.co_consts)[:650])

print("\nEARLY IDENTIFICATION NOTES")
lines282 = Path("/work/notes.md").read_text().splitlines()
selected282 = [s for s in lines282
               if re.search(r"identif|autoregress|instrument|randomiz|true effect|effect.*estimat|persistence|rho|lag coefficient", s, re.I)]
for s in selected282[:12]:
    print(s[:750])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 282; forwards 24. Architecture confirms generated-value feedback through recurrence and last-value registers. Run281 output truncated; recovering bounded helper and identification inventory. No weights changed.\n")
```
