The last result shows that two latent coordinates can be freed while nearly preserving their original contribution. It does **not** yet show that repurposing an attention head is safe or that the repair produces the right drug effects. I’ll inspect the architecture and retained calibration objects before constructing that pathway. It is viable only if it transports dose changes reliably without materially changing factual predictions.

```python
import os, re, inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== RECENT KERNEL OBJECTS ===")
_snapshot197 = list(globals().items())
for _name197, _obj197 in _snapshot197[-110:]:
    if _name197.startswith("_"):
        continue
    _shape197 = getattr(_obj197, "shape", None)
    if isinstance(_obj197, dict):
        print(_name197, "dict", "keys:", list(_obj197)[:18])
    elif callable(_obj197):
        try:
            print(_name197, "callable", inspect.signature(_obj197))
        except Exception:
            print(_name197, "callable")
    else:
        print(_name197, type(_obj197).__name__, _shape197)

print("\n=== EARLIER CAUSAL-EVIDENCE NOTES ===")
_notes197 = open("/work/notes.md").read()
_lines197 = _notes197.splitlines()
_hits197 = [
    i for i, s in enumerate(_lines197)
    if re.search(r"instrument|identif|confound|causal|lag.*estimate|slope|effect.*estimate", s, re.I)
]
for _i197 in _hits197[:35]:
    print("\n".join(_lines197[max(0, _i197-1):min(len(_lines197), _i197+3)]))

print("\nActive edits:", edited())
```
