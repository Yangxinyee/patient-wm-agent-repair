The aligned calibration test showed essentially no NLL improvement, so temperature scaling will not provide useful headroom. I’ll recover the architecture, record-based effect estimates, and validated edits before choosing another repair. A promising repair must improve the dose predictions without exceeding the 0.005 factual-NLL allowance.

```python
from pathlib import Path
import re, numpy as np, pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

notes208 = Path("/work/notes.md").read_text()
print("\n=== NOTEBOOK SIZE AND EARLY EVIDENCE ===", len(notes208))
print(notes208[:22000])
print("\n=== RELEVANT NOTEBOOK SECTIONS ===")
sections208 = re.split(r"(?=^## )", notes208, flags=re.M)
selected208 = [
    s for s in sections208
    if re.search(r"(Run[s ]*(?:7[5-9]|80|1[5-8][0-9])\b|causal|true effect|target effect|Forward(?:s)? ?(?:1[5-9]|2[0-3])\b)", s, re.I)
]
print("\n".join(s[:4500] for s in selected208)[-31000:])
print("\n=== SAVED BASELINE SCHEMA ===")
with np.load("/work/baseline77.npz") as z:
    print({k: (z[k].shape, str(z[k].dtype)) for k in z.files})
print("\n=== PERSISTENT HELPERS AND IMPORTANT OBJECTS ===")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        print("FUNCTION", k)
    elif re.search(r"(target|effect|causal|baseline|delta|direction|coef|dose|train_meta|contrast|feature|ridge|detector)", k, re.I):
        sh = getattr(v, "shape", None)
        if sh is not None:
            print(k, type(v).__name__, sh)
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:15])
        elif isinstance(v, (float, int, str)):
            print(k, repr(v)[:300])
print("\nCURRENT EDITS:", edited())
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 208: restarted-context recovery\n"
            "Aligned temperature calibration offers no meaningful factual headroom. "
            "Recovering architecture, causal evidence, validated frontier, and persistent helpers "
            "before proposing another edit. Baseline77 remains installed. "
            "Python runs: 208; forwards: 23.\n")
```
