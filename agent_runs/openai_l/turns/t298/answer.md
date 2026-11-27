The architecture and current edits are confirmed, but the output was truncated before the candidate-fitting objects appeared. I’ll inspect those objects compactly, without changing weights. The next step is to verify that the proposed candidate can be reproduced exactly before testing it.

```python
import inspect, re
from pathlib import Path

snap265 = dict(globals())
lines265 = ["=== RECENT OBJECTS ==="]
for k, v in snap265.items():
    if re.search(r"26[0-3]|25[5-9]|grid242|backup248|fitmask", k, re.I):
        if hasattr(v, "shape"):
            desc = f"{type(v).__name__} shape={tuple(v.shape)}"
        elif isinstance(v, dict):
            desc = f"dict len={len(v)} keys={repr(list(v)[:18])}"
        elif isinstance(v, (list, tuple)):
            desc = f"{type(v).__name__} len={len(v)} {repr(v[:3])[:260]}"
        elif inspect.isfunction(v):
            desc = "function " + str(inspect.signature(v))
        else:
            desc = repr(v)[:180]
        lines265.append(k + ": " + desc)

lines265.append("=== FITTING FUNCTION SIGNATURES ===")
for k, v in snap265.items():
    if inspect.isfunction(v) and re.search(
        r"fit|eval|head|candidate|pair|decoder|metric|nll|build|solve|stage", k, re.I
    ):
        lines265.append(k + str(inspect.signature(v)))

lines265.append("=== LAST 45 GLOBAL NAMES ===")
lines265.append(", ".join(list(snap265)[-45:]))
catalog265 = "\n".join(lines265)
print(catalog265[:9000])

notes265 = Path("/work/notes.md").read_text()
evidence265 = [
    s for s in re.split(r"(?=^## )", notes265, flags=re.M)
    if any(t in s.lower() for t in
           ["accepted records", "rejected early", "strengthened record", "causal timing"])
]
print("=== RECORD-EVIDENCE SUMMARY ===")
print("\n".join(evidence265)[-1800:])
```