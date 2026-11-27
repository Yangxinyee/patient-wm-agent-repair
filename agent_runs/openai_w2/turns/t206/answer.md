Across 199 stays, the factual NLL increase is **0.00342**, with appreciable sampling uncertainty. I’ll inspect the independent effect checks and persistence fits before changing weights again. Consistent estimates across methods would support stronger responses; disagreement would favor the validated conservative edit.

```python
import numpy as np
import pandas as pd

print("RECORD-ONLY EFFECT CHECKS")
budget178 = 6200
for section in sections177:
    header = section.splitlines()[0]
    if any(tag in header for tag in (
        "Run 6:", "Run 10:", "Run 11:", "Run 39:"
    )):
        shown = section[:min(len(section), budget178)]
        print(shown)
        budget178 -= len(shown)
        if budget178 <= 0:
            break

print("\nPERSISTENT STATISTICAL STATE-MODEL PARAMETERS")
parameter_names178 = []
for name, obj in list(globals().items()):
    if isinstance(obj, dict) and "rho" in obj and "b" in obj:
        parameter_names178.append(name)
        summary = {}
        for key in ("rho", "b", "Q", "R", "drift", "bias"):
            if key in obj:
                summary[key] = obj[key]
        print(name, summary)

print("\nSTATE-MODEL / TARGET FUNCTIONS")
for name, obj in list(globals().items()):
    if callable(obj) and any(tag in name.lower() for tag in (
        "kalman", "smooth", "state_model", "latent", "target", "fit_gauss",
        "ssm", "em_", "fit_sensor"
    )):
        code = getattr(obj, "__code__", None)
        print(name, "arguments/local names:",
              code.co_varnames[:20] if code else "(not a Python function)")

print("\nPanel index and sample:")
print(panel.index.names, panel.shape)
print(panel.head(4).to_string())
print("\nTuning panel sample:")
print(tune_panel.head(4).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run178: record-effect audit, no weight changes\n"
            "Reviewing threshold-IV and stay-crossfitted effect estimates "
            "together with the fitted latent persistence models. The "
            "199-stay factual budget passes on its point estimate but "
            "sampling uncertainty remains; target estimates are not truth.\n")
```