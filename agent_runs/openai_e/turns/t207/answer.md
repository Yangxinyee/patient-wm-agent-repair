```python
from pathlib import Path
import re, numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes178 = Path("/work/notes.md").read_text()
sections178 = re.split(r"(?=^## )", notes178, flags=re.M)
print("\nEARLY RECORD-BASED EVIDENCE")
print(notes178[:6500])
print("\nEDIT/TARGET RECOVERY")
for section in sections178:
    title = section.split("\n", 1)[0]
    if re.search(r"\b(?:58|63|70|71|72|73|74|75|76|77|78|79|80|81|82|83|107|148)\b", title):
        print(section[:1800])

print("\nSAVED BASELINE")
with np.load("/work/baseline77.npz") as a178:
    print([(k, a178[k].shape, str(a178[k].dtype)) for k in a178.files])
print("\nKERNEL OBJECTS (names and shapes only)")
for key178 in sorted(list(globals())):
    if re.search(r"(target|effect|causal|coef|baseline|insulin|ne_|Bdir|Binput|fit|ids|stay|train|Worig|Wbase|delta)", key178, re.I):
        obj178 = globals()[key178]
        if callable(obj178):
            print(key178, "callable")
        elif hasattr(obj178, "shape"):
            print(key178, type(obj178).__name__, tuple(obj178.shape))
        elif isinstance(obj178, dict):
            print(key178, "dict keys", list(obj178.keys())[:12])
        elif isinstance(obj178, (list, tuple)):
            print(key178, type(obj178).__name__, "length", len(obj178))
print("\nCURRENT EDITS", edited())

with Path("/work/notes.md").open("a") as f178:
    f178.write("\n## Run 178: recover causal targets and accepted baseline\n"
               "Insulin-only baseline77 remains installed. No new weights changed. "
               "Recovering architecture, early record-based causal evidence, and exact saved edit "
               "before a smaller factual-budgeted joint repair. Python runs: 178; forwards: 22.\n")
```
