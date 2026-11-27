The saved nonlinear screen contains held-out results, but no candidate has yet been installed or rollout-tested. I’ll rank those results and recover their feature definition. A selective doubling gain with acceptable factual loss would justify a trial; otherwise I’ll retain the simpler edit.

```python
import numpy as np
import pandas as pd
import inspect
import dis
from pathlib import Path

print("=== NONLINEAR HELD-OUT FRONTIER ===")
cols252 = ["key", "cv_dNLL", "tune_dNLL", "cv_A2.0", "cv_A0.5",
           "cv_C2.0", "cv_C0.5", "min_derivative", "max_derivative"]
print(frontier237[cols252].sort_values("cv_A2.0", ascending=False)
      .head(7).to_string(index=False))

print("\n=== NONLINEAR PACKS ===")
for key252 in ["train", "tune", ("norepinephrine", 0.5), ("norepinephrine", 2.0)]:
    pack252 = curve236[key252]
    print(key252, [(k, getattr(v, "shape", type(v).__name__))
                   for k, v in pack252.items()])
    for sub252 in ["mc", "pred"]:
        if sub252 in pack252 and isinstance(pack252[sub252], dict):
            print(" ", sub252, [(k, getattr(v, "shape", type(v).__name__))
                               for k, v in pack252[sub252].items()])

print("\n=== CACHE NAMES ===")
print([k for k, v in sorted(list(globals().items()))
       if isinstance(v, dict) and "inputs" in v and "target" in v])
print("\n=== FEATURE / CURVE OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or not any(t in k.lower()
                                    for t in ["knot", "hinge", "phi", "curve", "signal", "scalar"]):
        continue
    if isinstance(v, np.ndarray):
        print(k, v.shape, "first", v.reshape(-1)[:5])
    elif isinstance(v, (int, float, str)):
        print(k, v)
    elif isinstance(v, dict):
        print(k, "keys", list(v)[:6])

print("\n=== HEAD EVALUATOR RETURN / GLOBALS ===")
print(head177_eval.__code__.co_names)
print([x for x in head177_eval.__code__.co_consts
       if isinstance(x, str) and len(x) < 80])

print("\n=== NONLINEAR EXPERIMENT NOTES ===")
print("\n".join(s[:1000] for s in sections249
                if s.split("\n")[0].startswith(("Runs 236", "Run 237", "Run 238"))))

with Path("/work/notes.md").open("a") as fh252:
    fh252.write(
        "\n## Run 252: nonlinear candidate ranking\n"
        "- No weights changed. Ranking saved nonlinear candidates by held-out"
        " MAP doubling response and checking exact cached feature definitions.\n"
        "- A candidate requires factual safety and actual paired rollout benefit;"
        " local sensitivity alone is insufficient.\n"
    )
```