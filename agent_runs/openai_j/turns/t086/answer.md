The architecture confirms that drug doses enter the shared recurrent state, so broad edits can disturb more than glucose or MAP predictions. The recovered estimates are still conditional associations, not established effects. I’ll retrieve the later identification results and cached calculations in a shorter output; agreement across stronger identification checks would justify a repair target.

```python
from pathlib import Path
import pandas as pd

s = Path("/work/notes.md").read_text()
print("=== IDENTIFICATION NOTES, CONTINUED ===")
print(s[4800:10200])

print("\n=== CACHED VARIABLE NAMES ===")
names_now = sorted(k for k in globals() if not k.startswith("_"))
print(", ".join(names_now)[:2300])

print("\n=== TARGET / IV TABLES ===")
for filename in ["kernel_targets.csv", "dynamic_iv.csv", "sensor_iv.csv"]:
    t = pd.read_csv(Path("/work") / filename)
    print(filename, t.shape)
    print(t.to_string(index=False)[:450])

print("\n=== RECENT FUNCTION SIGNATURES ===")
for k in names_now:
    obj = globals()[k]
    if callable(obj) and getattr(obj, "__module__", "") == "__main__":
        c = getattr(obj, "__code__", None)
        if c is not None:
            print(k, c.co_varnames[:c.co_argcount])
```