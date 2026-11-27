The architecture confirms that the edits affect the reading head, not the recurrent dynamics. The records also show strong dose–condition confounding, so raw correlations are not credible effect estimates. The recovery output was truncated; I’ll retrieve only the causal conclusions and candidate interfaces needed for the next test.

```python
import inspect, re
from pathlib import Path

txt = Path("/work/notes.md").read_text()
secs = re.split(r"(?=^## )", txt, flags=re.M)
matched = [
    s for s in secs
    if re.search(
        r"identification|instrumental|causal target|record-derived target|"
        r"long.horizon|provisional target|true effects|target effects",
        s, re.I
    )
]
print("=== RELEVANT SECTION HEADINGS ===")
for s in matched:
    print(s.splitlines()[0])

print("\n=== RECENT EFFECT-ESTIMATION CONCLUSIONS ===")
for s in matched[-4:]:
    print(s[:1300])

print("\n=== RECENT FUNCTIONS ===")
for k, v in sorted(list(globals().items())):
    if inspect.isfunction(v) and re.match(r"^(r23|r24|c21|c22|c23|c24)", k):
        print(k, inspect.getfullargspec(v).args)

print("\n=== KEY OBJECTS ===")
for k in ["r235_safe", "r244_current", "r246_base", "r246_rows",
          "r246_results", "r240_results", "r240_models", "c210_train",
          "c210_tune", "r247_before"]:
    if k not in globals():
        continue
    v = globals()[k]
    if isinstance(v, dict):
        print(k, {str(a): str(getattr(b, "shape", type(b).__name__))
                  for a, b in v.items()})
    elif hasattr(v, "shape"):
        print(k, v.shape)
        if getattr(v, "ndim", 0) == 2 and v.shape[0] < 30:
            print(str(v)[:800])
    else:
        print(k, type(v).__name__)

print("\n=== OTHER CANDIDATE VARIABLE NAMES ===")
print([k for k in globals()
       if re.match(r"^(r240|r242|r244|r246|r247|c240)", k)])

print("\n=== AVAILABLE TUNING STAYS ===")
import pandas as pd
tuning_stay_ids = sorted(pd.read_csv(
    "/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique().tolist())
print("count", len(tuning_stay_ids),
      "first", tuning_stay_ids[:5], "last", tuning_stay_ids[-5:])
for name in ["c210_train", "c210_tune"]:
    if name in globals():
        ids = sorted(globals()[name]["meta"]["stay_id"].unique().tolist())
        print(name, "stays", len(ids), "first", ids[:3], "last", ids[-3:])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 250: bounded recovery\n"
            "Previous recovery output was truncated. Retrieved compact causal "
            "section headings and recent candidate interfaces. No weights changed.\n")
```
