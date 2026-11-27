The latest rollout strengthens the insulin response substantially, but the norepinephrine response improves only modestly. That alone does not establish accuracy or acceptable factual fit. I’ll recover the record-based effect estimates and pass 27’s factual results; those will determine whether this trial is worth retaining.

```python
from pathlib import Path
import numpy as np, pandas as pd, inspect, types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

resume_notes = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION NOTES ===")
print(resume_notes[:7000])

print("\n=== FORWARD 27 FACTUAL RESULTS ===")
for p in sorted(Path("/work/fwd_27").glob("*")):
    print(p.name)
    if p.suffix == ".csv":
        tab = pd.read_csv(p)
        print("shape:", tab.shape, "columns:", list(tab.columns))
        print(tab.head(2).to_string(index=False))
        print("NLL means:", {
            c: float(pd.to_numeric(tab[c], errors="coerce").mean())
            for c in tab.columns if "nll" in c.lower() or "loss" in c.lower()
        })
    elif p.suffix == ".json":
        print(p.read_text()[:2500])

print("\n=== RETAINED HELPER INTERFACES ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, types.FunctionType) and obj.__module__ == "__main__":
        try:
            print(name, str(inspect.signature(obj)))
        except Exception:
            print(name)

print("\n=== CANDIDATE ARCHIVES ===")
for fn in ["recurrent_glucose_valid.npz", "map_feature_trial_backup.npz",
           "map_regularized_trial.npz", "glucose_115_candidate.npz"]:
    with np.load("/work/" + fn, allow_pickle=False) as arc:
        print(fn, [(k, arc[k].shape) for k in arc.files])
        for k in arc.files:
            if arc[k].size <= 12:
                print(" ", k, arc[k])

print("\n=== RELEVANT RETAINED OBJECTS ===")
tokens = ("target", "route", "fisher", "rec_", "regular", "trial", "valid",
          "jac", "case", "feature", "glucose", "mapfit", "b0", "bin")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or not any(t in name.lower() for t in tokens):
        continue
    if isinstance(obj, np.ndarray):
        print(name, "array", obj.shape)
    elif isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, list(obj.columns)[:8])
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj)[:8])
    elif isinstance(obj, (str, int, float, bool, tuple, list)) and len(repr(obj)) < 250:
        print(name, repr(obj))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 282: recover final calibration evidence\n"
        "- Forward 28 current trial: insulin x2/half -11.952/+5.366; NE x2/half +0.665/-0.164.\n"
        "- These are predictions, not drug-effect truth. Recovering record targets and forward 27 factual fit before retaining the MAP addition.\n"
        "- No weights changed. Python runs: 282; forward passes: 28.\n"
    )
```