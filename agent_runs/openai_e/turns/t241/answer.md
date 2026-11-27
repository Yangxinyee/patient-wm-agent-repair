The architecture confirms that doses affect the recurrent state, while predictions also depend strongly on previous readings. The early record analysis shows substantial confounding, so raw dose–outcome associations are not reliable effect estimates. The output was truncated before the later findings; I’ll recover those compactly. Agreement between the identification checks and rollout validation would support the existing repair.

```python
import re, numpy as np
from pathlib import Path

notes209 = Path("/work/notes.md").read_text()
secs209 = re.split(r"(?=^## )", notes209, flags=re.M)
wanted209 = []
for s in secs209:
    heading = s.splitlines()[0] if s.splitlines() else ""
    if re.search(r"Run[s ]*(?:7[5-9]|80|107|148|15[0-9]|17[5-9]|18[0-9]|202)\b", heading):
        wanted209.append(s[:850])
print("=== TARGETED FINDINGS ===")
print("\n".join(wanted209)[-5800:])

print("\n=== BASELINE ARCHIVE ===")
with np.load("/work/baseline77.npz") as z:
    for k in z.files:
        a = z[k]
        print(k, a.shape, "norm", float(np.linalg.norm(a)), 
              "value", a.tolist() if a.size <= 6 else "")

print("\n=== RECENT OBJECTS ===")
for k in sorted(list(globals())):
    if re.search(r"(?:20[0-7]|19[7-9]|17[5-9]|82|77)$", k):
        v = globals()[k]
        if callable(v):
            print("FUNCTION", k, getattr(getattr(v, "__code__", None), "co_varnames", ())[:10])
        elif isinstance(v, dict):
            print(k, "DICT", list(v)[:8])
        elif hasattr(v, "shape"):
            print(k, type(v).__name__, v.shape)
        elif isinstance(v, (float, int, str)) and len(str(v)) < 180:
            print(k, repr(v))

print("\n=== KEY HELPERS ===")
for k in sorted(list(globals())):
    v = globals()[k]
    if callable(v) and getattr(v, "__module__", "") == "__main__" and re.search(
        r"(build|decode|simulate|roll|panel|head|feature|state|embed|fit|install|metric|eval)", k, re.I
    ):
        print(k, getattr(getattr(v, "__code__", None), "co_varnames", ())[:8])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 209: compact context recovery\n"
            "Run208 output truncated before later evidence. Recovering targeted causal/validation "
            "findings, baseline archive schema, and recent fitting helpers with bounded output. "
            "No weights changed. Python runs: 209; forwards: 23.\n")
```
