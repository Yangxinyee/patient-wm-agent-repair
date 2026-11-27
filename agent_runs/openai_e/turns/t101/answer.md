Restoring the baseline isolates the problem: before the norepinephrine addition, insulin shifts were −12.71 and +4.33; afterward they were −12.51 and +3.33. I’ll inspect the retained feature fits and local forward helpers to build a less leaky correction. Lower factual activation and negligible insulin contrasts would support another trial.

```python
import inspect
from pathlib import Path
import numpy as np

report86 = []
def add86(title, value, limit=1600):
    report86.append(title + "\n" + str(value)[:limit])

for name86 in ["prelude46", "core_update_local", "next_means45",
               "edited_action46", "feature_metrics57", "array80"]:
    f86 = globals().get(name86)
    if callable(f86):
        try:
            src86 = inspect.getsource(f86)
        except Exception:
            src86 = ("signature=" + str(inspect.signature(f86)) +
                     "\nvariables=" + str(f86.__code__.co_varnames) +
                     "\nreferences=" + str(f86.__code__.co_names))
        add86(name86, src86, 1900)

add86("RUN 68 NOTE",
      "\n".join(s for s in sections85 if s.startswith("## Run 68")), 1100)

names86 = []
for k86, v86 in sorted(list(globals().items())):
    if isinstance(v86, np.ndarray):
        if v86.ndim >= 2 and (192 in v86.shape or 188 in v86.shape):
            names86.append((k86, v86.shape))
    elif isinstance(v86, dict) and any(s in k86.lower() for s in
                                     ["feature", "fit", "data", "contrast", "e46", "states", "direction"]):
        names86.append((k86, "dict", list(v86)[:6]))
add86("FEATURE/STATE OBJECTS", names86, 2500)

for name86 in ["metrics82", "metrics81", "fits82", "features82", "ne_metrics82",
               "ne_fits82", "direction82", "desired82", "check82"]:
    if name86 in globals():
        v86 = globals()[name86]
        if isinstance(v86, dict):
            add86(name86, v86, 1600)
        elif isinstance(v86, np.ndarray):
            add86(name86, {"shape": v86.shape, "norm": float(np.linalg.norm(v86)),
                          "first": v86.ravel()[:12].tolist()}, 600)

print("\n\n".join(report86)[:9200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 86: NE interference and helper recovery\n"
            "Baseline forward10 insulin shifts -12.708/+4.335, NE +.099/-1.004. "
            "NE trial reduced insulin-half response by ~1.0 and failed factual fit. "
            "Insulin-only baseline remains installed. Recovering local forward and constrained regression "
            "objects to improve detector selectivity and doubling/halving balance. "
            "Python runs: 86; forwards: 12.\n")
```