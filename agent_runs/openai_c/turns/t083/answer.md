The numeric edit changes drug-related states, but its insulin effect remains far below the record-based estimate, and norepinephrine’s response is still near zero. I’ll compare the rollout time courses with the estimated persistence. Effects that appear initially but fail to accumulate would support testing the model’s reading-feedback pathway rather than increasing dose gain again.

```python
import pandas as pd, numpy as np, re
from pathlib import Path

print("=== ROLLOUT TIME COURSES ===")
for directory, label, model in [
    ("/work/fwd_2", "original", "original"),
    ("/work/fwd_9", "numeric gain 4", "edited"),
]:
    path = Path(directory) / f"dose_edit_{model}.csv"
    if not path.exists():
        print(label, "file missing")
        continue
    rr = pd.read_csv(path)
    print(label, "columns", list(rr.columns))
    if {"hour", "edit", "shift"}.issubset(rr.columns):
        rr = rr[(rr.hour >= 13) & (rr.hour <= 24)]
        print(rr.groupby(["hour", "edit"]).shift.mean().unstack().round(3).to_string())

print("\n=== PERSISTENCE / TARGET VARIABLES ===")
selected_names = [n for n in sorted(globals())
                  if any(t in n.lower() for t in ["persist", "target", "probe", "ar_", "dynamic"])]
print("names:", selected_names)
for n in selected_names:
    v = globals()[n]
    if isinstance(v, dict) and n not in ["dynamic_moments", "q_probe"]:
        print(n, repr(v)[:900])
    elif isinstance(v, (float, int, str)):
        print(n, v)

print("\n=== READING FEEDBACK AND PRELUDE TENSORS ===")
for n in NAMES:
    if n.startswith("last_numeric.") or n.startswith("core.prelude.") or n.startswith("core.coda."):
        print(n, SHAPES[n])

print("\n=== EARLIER EDIT DESIGN NOTES ===")
text = Path("/work/notes.md").read_text()
for pattern in [r"## Run 6", r"## Run 7", r"## Run 8", r"## Python run 12",
                r"## Python run 13", r"## Python run 23"]:
    match = re.search(pattern, text)
    if match:
        print(text[match.start():match.start()+650])
print("\n=== CANDIDATE OP COUNTS ===")
for k, v in repair_candidates.items():
    print(k, type(v).__name__, len(v))
print("active:", ops())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 57\n"
            "- Paired states show numeric gain alters z more strongly than normalized out.\n"
            "- Record-based immediate slopes remain about -1.93 insulin and +3.1 norepinephrine.\n"
            "- Inspecting rollout accumulation and reading-feedback pathways before another pilot.\n"
            "- No edits changed; forward passes used: 24.\n")
```