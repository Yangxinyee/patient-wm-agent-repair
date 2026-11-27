The recovered records support a negative insulin effect and a positive norepinephrine effect, conditional on the recorded-history adjustment. Panel doses are raw rates; norepinephrine’s recorded total dose uses a separate scale. I’ll now check the longitudinal kernels and repair construction. Their units and eligible-hour aggregation must agree before they can guide calibration.

```python
import re
import numpy as np
from pathlib import Path

g254 = dict(globals())
for n in ("kernels137", "provisional_kernels", "boot_kernels137",
          "r244_candidates", "r244_results", "r249_key"):
    v = g254.get(n)
    print("\nOBJECT", n, type(v).__name__)
    if isinstance(v, dict):
        for k, x in list(v.items())[:12]:
            if isinstance(x, dict):
                print(k, "keys", list(x.keys())[:16])
                for kk in ("kernel", "response", "se", "dose_cov"):
                    if kk in x:
                        print(" ", kk, np.asarray(x[kk]).round(5))
            elif hasattr(x, "shape"):
                a = np.asarray(x)
                print(k, a.shape, a.round(5) if a.size <= 40 else "")
            else:
                print(k, repr(x)[:350])
    elif hasattr(v, "shape"):
        a = np.asarray(v)
        print(a.shape, a.round(5) if a.size <= 40 else "")
    else:
        print(repr(v)[:500])

for n in ("fit_lp136", "deconvolve_lp136", "factual_eval194"):
    fn254 = g254.get(n)
    if callable(fn254):
        print("\nFUNCTION", n)
        print("arguments", fn254.__code__.co_varnames[:fn254.__code__.co_argcount])
        print("referenced names", fn254.__code__.co_names)

sections254 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
wanted254 = (94, 95, 136, 137, 138, 185, 186, 194)
print("\nSELECTED NOTEBOOK SECTIONS")
for run254 in wanted254:
    for s254 in sections254:
        if re.match(rf"## Run {run254}(?:\D|$)", s254):
            print(s254[:720])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 254\n"
            "Raw-rate panel units confirmed (insulin 0–6, NE 0–1); "
            "NE recorded total dose is separately normalized. Earlier RD and "
            "history-adjusted estimates support insulin negative/NE positive effects, "
            "subject to identification assumptions. Recovering lag kernels and "
            "response-channel interfaces. No weights changed.\n")
```