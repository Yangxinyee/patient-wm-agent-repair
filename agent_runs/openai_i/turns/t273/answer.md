The sharper candidate is installed, but it is not validated. I’ll recover the exact tuning-stay IDs used in the failed test so the next comparison uses the same records. Passing that matched factual check would justify testing rollouts; exceeding +0.005 would require rejection.

```python
import json
import pandas as pd

folders244 = sorted(
    [p for p in Path("/work").glob("fwd_*") if p.is_dir()],
    key=lambda p: int(p.name.split("_")[-1])
)
for folder244 in reversed(folders244):
    path244 = folder244 / "factual_original.csv"
    if path244.exists():
        original244 = pd.read_csv(path244)
        print("Reference:", path244)
        print("Columns:", list(original244.columns))
        staycol244 = next(
            c for c in original244.columns
            if c in ["stay_id", "stay", "id"] or "stay" in c.lower()
        )
        stays244 = [
            int(x) for x in original244[staycol244].drop_duplicates().tolist()
        ][:100]
        print("Matched factual request:")
        print(json.dumps({"type": "factual", "model": "both",
                          "split": "tuning", "stays": stays244}))
        break

with open("/work/notes.md", "a") as f244:
    f244.write("\n## Python 244 — prepare matched GPU factual validation\n"
               "- Sharp243 remains provisional.\n"
               "- Recovering the exact tuning stays used for the failed response233 comparison.\n"
               "- Acceptance requires edited-minus-original value NLL <= .005.\n")
```
