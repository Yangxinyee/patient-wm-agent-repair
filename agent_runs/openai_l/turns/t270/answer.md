The fit cache contains stronger candidates, so we can reuse the analytical fits rather than rebuild them. I’ll compare their factual penalties and dose slopes, and recover the accepted effect estimates that were cut off in the last output. A stronger candidate is useful only if it improves response without spending too much factual-fit margin.

```python
import numpy as np
from pathlib import Path

print("BROAD-ANCHOR NORMAL EQUATIONS:")
for key, st in stats234.items():
    print("KEY", key)
    for k, v in st.items():
        if k == "arch":
            print(k, v["style"], v["lam"], "units", v["units"])
        elif hasattr(v, "shape"):
            print(k, v.shape, "norm", round(float(np.linalg.norm(v)), 5))
        elif isinstance(v, dict):
            print(k, {str(q): (getattr(z, "shape", None),
                               round(float(np.linalg.norm(z)), 5)
                               if hasattr(z, "shape") else repr(z)[:50])
                      for q, z in v.items()})
        else:
            print(k, repr(v)[:150])

print("\nEXISTING CANDIDATE SCORES:")
for key, v in expanded234.items():
    print(key, "NLL deltas",
          np.round([v["train_delta"], v["val_delta"], v["tune_delta"]], 6),
          "slopes", np.round(v["slopes"], 3))

print("\nMAKE_ARCH CONSTANTS:", make_arch215.__code__.co_consts)
print("\nACCEPTED RECORD-EFFECT EVIDENCE:")
print(Path("/work/notes.md").read_text()[5500:10800])
```