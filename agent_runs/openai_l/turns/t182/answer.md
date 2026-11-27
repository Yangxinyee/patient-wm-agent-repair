The new write is locally quiet, but that does not establish factual safety: earlier quiet-looking recurrent edits failed the full-record test. I’ll recover the architecture and audit stay IDs before testing this candidate. A paired NLL increase below 0.005 is necessary to keep it; a larger increase would reject it.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

print("\n=== EARLY NOTEBOOK / CAUSAL EVIDENCE ===")
notes168 = Path("/work/notes.md").read_text()
print(notes168[:14000])

print("\n=== PRIOR FACTUAL AUDIT ===")
audit168 = pd.read_csv("/work/fwd_8/factual_original.csv")
print("columns:", list(audit168.columns))
staycol168 = next(c for c in audit168.columns if "stay" in c.lower())
audit_stays168 = audit168[staycol168].drop_duplicates().tolist()[:100]
print("audit_stays168 =", audit_stays168)

print("\n=== RELEVANT PERSISTENT OBJECTS ===")
for name in sorted(list(globals())):
    if any(s in name for s in (
        "167", "145", "152", "118", "125", "kernel", "lag",
        "head63", "core63", "step63", "forward63", "capture"
    )):
        obj = globals()[name]
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif callable(obj):
            print(name, "callable", getattr(obj, "__code__", None).co_varnames
                  if hasattr(obj, "__code__") else "")
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:20])
        elif isinstance(obj, (float, int, str)):
            print(name, repr(obj)[:200])

print("\nCURRENT EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 168 / restart recovery\n"
            "One-step b_in write remains unvalidated. Recovering architecture, "
            "causal evidence, and the existing 100-stay audit set before the "
            "paired factual test. No additional weights changed.\n")
```