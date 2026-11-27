The last audit matched no MAP readings, so its NaNs tell us nothing about drug effects. I’ll check the matching keys and recover the causal targets and model interfaces before changing weights. Successful matching will make the diagnostic interpretable; the insulin-only edit remains in place.

```python
from pathlib import Path
import pandas as pd, numpy as np, re, inspect

print("ARCHITECTURE\n", Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA DESCRIPTION\n", Path("/data/README.md").read_text())

tuning97 = pd.read_csv("/data/tuning.csv")
print("\nTUNING", tuning97.shape, tuning97.dtypes.to_dict())
print(tuning97.head(8).to_string(index=False))

for folder97 in ("/work/fwd_9", "/work/fwd_10", "/work/fwd_11"):
    for path97 in sorted(Path(folder97).glob("*.csv")):
        sample97 = pd.read_csv(path97)
        print("\nFORWARD FILE", str(path97), sample97.shape)
        print(sample97.head(3).to_string(index=False))

notes97 = Path("/work/notes.md").read_text()
sections97 = re.split(r"(?=^## )", notes97, flags=re.M)
print("\nNOTEBOOK INTRODUCTION\n", sections97[0][:5000])
print("\nSELECTED EARLIER FINDINGS")
for section97 in sections97[1:]:
    heading97 = section97.splitlines()[0]
    if any(s97 in heading97.lower() for s97 in
           ("causal", "instrument", "insulin", "baseline", "run 69:", "run 70:", "run 71:")):
        print(section97[:2200])

print("\nRELEVANT HELPERS AND ARRAYS")
for name97, obj97 in list(globals().items()):
    if not name97.startswith("_") and (
        callable(obj97) and any(s97 in name97.lower() for s97 in
                               ("core", "poster", "encode", "feature", "forward", "head", "predict"))
        or name97.endswith(("94", "95", "96"))
    ):
        if callable(obj97):
            try:
                print(name97, inspect.signature(obj97))
            except Exception:
                print(name97, type(obj97).__name__)
        elif hasattr(obj97, "shape"):
            print(name97, type(obj97).__name__, obj97.shape)
        elif isinstance(obj97, dict):
            print(name97, "dict keys", list(obj97)[:12])

with open("/work/notes.md", "a") as f97:
    f97.write("\n## Run 97: recover targets and fix empty residual audit\n"
              "Run96 matched zero future MAP readings; its NaNs are not evidence of a null effect. "
              "Inspecting record/prediction keys, causal estimates and model interfaces. "
              "No weights changed; baseline77 remains installed. Python runs: 97; forwards: 12.\n")
```