The last result confirms that the saved baseline is exactly the validated insulin-only edit. The records suggest norepinephrine should raise MAP, while the model’s rollout barely responds. I’ll recover the update mechanics and constrained fitting objects before constructing a separate correction. The hypothesis will be supported only if both norepinephrine rollouts improve while insulin responses and factual fit remain acceptable.

```python
from pathlib import Path
import re, inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes78 = Path("/work/notes.md").read_text()
print("\n=== EDIT DEVELOPMENT: RUNS 58–64 ===")
for run78 in range(58, 65):
    match78 = re.search(rf"(?m)^## Run {run78}:.*?(?=^## |\Z)", notes78, re.S)
    if match78:
        print(match78.group(0)[:1700])

print("\n=== RETAINED FITS AND HELPERS ===")
keys78 = [
    "W0", "core_lin", "core_layer", "core_norm", "core_flow",
    "coda46", "prelude46", "next_means45",
    "initial_matrix56", "null_basis57", "fit_drug57", "targets57",
    "slot_fits47", "slot_deltas47", "slot_targets47",
    "bin_directions46", "J46", "direction46",
    "candidate58", "insulin_feature58", "direction58",
    "fwd10", "ids", "tuning_ids"
]
for key78 in keys78:
    if key78 not in globals():
        continue
    obj78 = globals()[key78]
    desc78 = type(obj78).__name__
    if hasattr(obj78, "shape"):
        desc78 += f" shape={tuple(obj78.shape)}"
    elif isinstance(obj78, dict):
        desc78 += f" keys={list(obj78)[:12]}"
    elif callable(obj78):
        try:
            desc78 += f" signature={inspect.signature(obj78)}"
        except Exception:
            pass
        if hasattr(obj78, "__code__"):
            desc78 += f" dependencies={obj78.__code__.co_names}"
    print(f"{key78}: {desc78}")

print("\n=== RELEVANT RUN 57–64 OBJECT NAMES ===")
for key78, obj78 in list(globals().items()):
    if re.search(r"(57|58|59|60|61|62|63|64)$", key78):
        if any(word78 in key78.lower() for word78 in
               ["fit", "feature", "coef", "direction", "candidate", "target",
                "basis", "jac", "null", "strength", "metric"]):
            desc78 = type(obj78).__name__
            if hasattr(obj78, "shape"):
                desc78 += f" {tuple(obj78.shape)}"
            elif isinstance(obj78, dict):
                desc78 += f" keys={list(obj78)[:10]}"
            print(key78, desc78)

with open("/work/notes.md", "a") as file78:
    file78.write(
        "\n## Run 78: restart recovery for NE surgery\n"
        "Baseline77 is preserved. Recovering architecture and constrained-feature "
        "objects before a reversible NE edit; no weights changed. "
        "Python runs: 78; forwards: 10.\n"
    )
```