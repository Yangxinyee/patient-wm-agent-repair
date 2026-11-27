The architecture confirms that a dose edit must pass through the shared recurrent state before reaching the value heads. The output was truncated before the audit IDs appeared, so I’ll retrieve those and the compact wire diagnostics. The next factual test will determine whether the locally quiet write is also safe over complete stays.

```python
import pandas as pd
import numpy as np
from pathlib import Path

audit169 = pd.read_csv("/work/fwd_8/factual_original.csv")
print("AUDIT COLUMNS:", list(audit169.columns))
staycol169 = next(c for c in audit169.columns if "stay" in c.lower())
audit_stays169 = audit169[staycol169].drop_duplicates().tolist()[:100]
print("AUDIT STAYS:", audit_stays169)

print("\nWIRE OBJECTS:")
for name in ["B167", "U167", "gain167", "wire_out152", "Wrepair125",
             "QE125", "SP125", "R118", "P125", "Jselected118",
             "Vsignal118", "chosen125", "Vnew125"]:
    if name in globals():
        obj = globals()[name]
        if hasattr(obj, "shape"):
            arr = np.asarray(obj.detach().cpu() if hasattr(obj, "detach") else obj)
            print(name, "shape", arr.shape, "norm", float(np.linalg.norm(arr)))
        else:
            print(name, repr(obj)[:700])

print("\nCPU HELPER SIGNATURES:")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and hasattr(obj, "__code__") and any(
        s in name.lower() for s in ["step", "head", "wire", "bundle", "prelude", "capture"]
    ):
        print(name, obj.__code__.co_varnames[:obj.__code__.co_argcount])

print("\nRECENT WIRE NOTES:")
sections169 = Path("/work/notes.md").read_text().split("\n## ")
for section in sections169:
    title = section.split("\n", 1)[0]
    if any(s in title for s in ["Run 110", "Run 111", "Run 112", "Run 113",
                               "Run 114", "Run 115", "Run 125", "Run 152",
                               "Run 153", "Run 154", "Run 155", "Run 167"]):
        print("##", section[:1500])

print("CURRENT EDITS:", edited())
```