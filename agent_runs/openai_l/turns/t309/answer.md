The rollout improves insulin further: doubling gives −10.337 and halving +5.389. Norepinephrine doubling improves to +1.830, but halving changes little from the original at −0.880. I’ll back up this passing candidate and review the persistence evidence and fitting interface. Stronger instantaneous slopes alone would not justify another edit.

```python
from pathlib import Path
import re, inspect
import numpy as np

backup273 = {n: load(n).clone() for n in edited()}
with Path("/work/notes.md").open("a") as f:
    f.write("""
## Run 273 / weaker broadened head validated
Current arch263 + grid269 (.25,.15), hash 58b516f6e0.
Forward22 (fwd_23), tuning stays100–199: factual delta +.00409
(edited1.61991 vs original1.61582), within tolerance.
Forward23 (fwd_24), same stays: shifts [-10.337,+5.389,+1.830,-.880],
original [-4.072,+2.273,+.615,-.858].
Insulin improves over prior .5/.4 head; NE half barely improves over original.
backup273 preserves all seven current tensors. Need assess persistence and
record-supported rollout scale, not equate immediate decoder slopes with success.
""")

text273 = Path("/work/notes.md").read_text()
print("=== CAUSAL DYNAMICS EVIDENCE ===")
for section in re.split(r"(?m)(?=^## )", text273):
    header = section.splitlines()[0] if section else ""
    if re.match(r"## Run (17|18|19|20|21|22|23|24|25|26)\b", header):
        print(section[:1900])

def describe273(x, depth=0):
    if isinstance(x, dict):
        return {str(k): describe273(v, depth+1) for k,v in list(x.items())[:12]} if depth < 2 else list(x.keys())[:12]
    if isinstance(x, (tuple, list)):
        return [describe273(v, depth+1) for v in x[:8]] if depth < 2 else f"{type(x).__name__}[{len(x)}]"
    if hasattr(x, "shape"):
        return f"{type(x).__name__}{tuple(x.shape)}"
    return str(x)[:180]

print("\n=== CURRENT ANALYTIC FIT OBJECTS ===")
for n in ["arch263", "grid269", "trial263", "headbase214", "T0",
          "GF", "GDI", "GDN", "RF", "RDI", "RDN", "feature_scale"]:
    if n in globals():
        x = globals()[n]
        print(n, describe273(x))
print("SELECTED_GRID_ENTRY", describe273(grid269[(.25,.15)]))
print("FIT_FUNCTIONS")
for n in sorted(list(globals())):
    x = globals()[n]
    if callable(x) and re.search(r"(234|242|260|261|262|263|269)$", n):
        try:
            print(n, inspect.signature(x), "locals", x.__code__.co_varnames,
                  "globals", x.__code__.co_names)
        except (TypeError, AttributeError):
            pass
```
