The second independent factual audit also passes, at +0.00369 NLL. That leaves some—but not much—room to strengthen norepinephrine. I’ll compare the existing analytic candidates before changing weights, looking for better MAP dose responses without sacrificing insulin or crossing the factual limit.

```python
from pathlib import Path
import numpy as np
import inspect, re

with Path("/work/notes.md").open("a") as f:
    f.write("""
## Forward24 / further independent factual validation
Current weaker broadened head (58b516f6e0), tuning stays200–299:
NLL1.61463 vs original1.61094, delta+.00369 over4563 readings.
Together with stays100–199 (+.00409), weighted independent delta~+.00389.
Current candidate remains saved and backed up in backup273.
""")

print("=== WEAKER-HEAD CANDIDATES: train / validation / tuning / slopes ===")
for key, value in grid269.items():
    print(key, value["score"])
print("\n=== STRONGER BROADENED CANDIDATES ===")
for key in [(0.3,0.3), (0.5,0.4), (0.5,0.7)]:
    print(key, trial263[key]["score"])

print("\n=== STATISTICS AND FIT INTERFACES ===")
for n in sorted(list(globals())):
    if re.search(r"stats.*(263|269)$|^(B179|B225|C179|fitmask179|valmask179|P|A|kernels19|ar2_predictions19)$", n):
        x = globals()[n]
        print(n, describe273(x))
        if hasattr(x, "columns"):
            print("columns", list(x.columns)[:35], "index", x.index.names)
print("solve269 constants", solve269.__code__.co_consts)
for n in ["build_stats263", "solve269", "score234", "nll179", "nll228"]:
    if n in globals():
        fun = globals()[n]
        print(n, inspect.signature(fun), "globals", fun.__code__.co_names)
```
