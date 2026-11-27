The exact record-based targets are **−12.96, +6.48, +4.28 and −2.14**. The current edit underestimates both norepinephrine responses as well as insulin halving. I’ll check whether individual stays show saturation or opposing effects, and inspect how dose information enters the state. That will guide the next reversible edit rather than another blind gain change.

```python
import inspect, numpy as np, pandas as pd, re
from pathlib import Path

r271 = pd.read_csv("/work/fwd_22/dose_edit_edited.csv")
o271 = pd.read_csv("/work/fwd_22/dose_edit_original.csv")
m271 = r271.merge(o271, on=["stay_id", "hour", "edit"], suffixes=("_edit", "_orig"))
m271["counter"] = m271["factual_edit"] + m271["shift_edit"]
m271["baseline_change"] = m271["factual_edit"] - m271["factual_orig"]
print("=== REACHED ROLLOUT LEVELS ===")
for label, g in m271[m271.reached_edit].groupby("edit"):
    print(label, "n", len(g), "positive_fraction", round((g.shift_edit > 0).mean(), 3))
    print(g[["factual_edit", "factual_orig", "counter", "shift_edit", "baseline_change"]]
          .quantile([0, .1, .5, .9, 1]).round(3).to_string())
    print("by_hour", g.groupby("hour")[["shift_edit", "baseline_change"]]
          .mean().round(2).to_dict())

print("=== LOCAL FUNCTION INVENTORY ===")
fn271 = [(n, o) for n, o in list(globals().items())
         if inspect.isfunction(o) and getattr(o, "__module__", "") == "__main__"]
print(" ".join(n for n, o in fn271))
print("=== DOSE / PRELUDE HELPERS ===")
remaining271 = 4700
for name in ["posterior153", "direct_posterior_diagnostic", "prelude155",
             "prelude_parts145", "prelude185", "prelude254", "normalize267"]:
    obj = globals().get(name)
    if inspect.isfunction(obj) and remaining271 > 0:
        try:
            src = inspect.getsource(obj)
        except Exception as e:
            src = str(e)
        text = src[:min(remaining271, 2200)]
        print(name, text)
        remaining271 -= len(text)

print("=== SAVED TEMPORAL-DIRECTION FINDINGS ===")
n271 = Path("/work/notes.md").read_text().splitlines()
for line in n271:
    if re.search(r"Python run (19[4-9]|20[0-9]|21[0-5]|22[01])\b", line):
        print(line[:650])

with Path("/work/notes.md").open("a") as f:
    f.write("\nPython run 271; forwards 22. Recovered exact reference shifts (-12.9555,+6.4778,+4.2797,-2.1398). Inspecting reached-row saturation, baseline drift, temporal-direction history and exact carrier helpers before choosing a reversible experiment. Current candidate unchanged.\n")
```
