The retained analysis contains multi-hour drug-response kernels, which are more relevant than the immediate regressions. The inspection stopped on a display error; no weights changed. I’ll now retrieve those kernels, their implied rollout targets, and the edit construction so the next experiment tests a specific mismatch.

```python
import inspect, numpy as np, pandas as pd
from pathlib import Path

print("=== RECORD-DERIVED KERNELS ===")
for name228 in ["causal_beta80", "provisional_kernels", "kernels137", "kernel_models88",
                "target_table89", "target_rows89", "target_rows88", "r225"]:
    if name228 not in globals():
        continue
    value228 = globals()[name228]
    print(name228)
    if isinstance(value228, pd.DataFrame):
        print(value228.head(12).to_string(index=False))
    elif isinstance(value228, dict):
        for key228, item228 in value228.items():
            text228 = repr(item228)
            print(" ", key228, text228[:1000])
    else:
        print(repr(value228)[:1000])

print("\n=== LOCAL FUNCTION INTERFACES ===")
for name228, value228 in list(globals().items()):
    if inspect.isfunction(value228) and getattr(value228, "__module__", "") == "__main__":
        try:
            print(name228, inspect.signature(value228))
        except Exception:
            pass

print("\n=== LATER CAUSAL / TARGET NOTES ===")
sections228 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
budget228 = 2300
for section228 in sections228:
    m228 = re.match(r"## Run (\d+)", section228)
    if m228 and 50 <= int(m228.group(1)) <= 150 and re.search(
            r"target|kernel|instrument|causal|effect", section228, re.I):
        excerpt228 = section228[:500]
        if len(excerpt228) > budget228:
            break
        print(excerpt228)
        budget228 -= len(excerpt228)

with open("/work/notes.md", "a") as f228:
    f228.write("\n## Runs 227–228\nRun 227 inspection stopped on a display-only size/type error; no edits occurred. Retrieved retained longitudinal kernels, targets and local function interfaces to choose a specific calibration experiment. No saved weights changed.\n")
```