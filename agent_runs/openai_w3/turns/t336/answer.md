The recovered notes show record-derived evidence for insulin lowering glucose and norepinephrine raising MAP, with effects persisting beyond one hour. The output was truncated before the editing helpers appeared, so I still cannot safely construct a drug-specific change. I’ll retrieve a compact inventory; it should show whether the existing correction can be adjusted separately for each drug.

```python
import re, inspect, numpy as np
from pathlib import Path

print("=== FUNCTIONS (bounded) ===")
fnames = sorted(k for k, v in list(globals().items())
                if inspect.isfunction(v) and not k.startswith("_"))
print(str(fnames)[:2500])

print("\n=== CHECKPOINTS / CANDIDATES ===")
rows = []
for k, v in sorted(list(globals().items())):
    if re.search(r"checkpoint|candidate|carrier|direction|gamma|replay|coef|beta|ids", k, re.I):
        if inspect.isfunction(v):
            s = f"function {inspect.signature(v)}"
        elif isinstance(v, dict):
            s = f"dict({len(v)}), keys={list(v)[:8]}"
        elif hasattr(v, "shape"):
            s = f"{type(v).__name__} {tuple(v.shape)}"
        elif isinstance(v, (list, tuple)):
            s = f"{type(v).__name__}({len(v)}) {repr(v[:3])[:100]}"
        else:
            s = f"{type(v).__name__} {repr(v)[:100]}"
        rows.append(f"{k}: {s}")
print("\n".join(rows)[:4000])

print("\n=== EARLIER NUMERICAL ROLLOUT RESULTS ===")
text = Path("/work/notes.md").read_text()
selected = []
for line in text.splitlines():
    if re.search(r"rollout|readouts|shifts", line, re.I) and re.search(r"\d+\.\d+", line):
        if not re.search(r"Python run(?:s)?:? 2[6-9]\d", line):
            selected.append(line)
print("\n".join(x[:300] for x in selected[-10:])[:3000])

print("\n=== PASS 29 FILES ===")
print([p.name for p in Path("/work/fwd_29").glob("*")])
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 291; forwards 30. Recovery output from run290 was truncated before helper inventory. Reading bounded retained-function/checkpoint inventory and earlier numerical validations. Record-derived signs and persistent effects are supported; no weights changed.\n")
```
