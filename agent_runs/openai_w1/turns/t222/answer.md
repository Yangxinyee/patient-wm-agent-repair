The recovered estimates imply substantial negative insulin effects and positive norepinephrine effects. The fivefold numeric edit changed internal slots much more than the output summary, which may explain its limited benefit. I’ll inspect the carrier arrays and saved rollout results, then test a reversible candidate.

```python
import pandas as pd

print("=== LATENT-CARRIER CACHE ===")
for k in sorted(k for k in list(globals()) if k.startswith("lc_")):
    print(k, brief198(globals()[k]))

print("\n=== EARLY NUMERIC-EDIT NOTES ===")
for s in selected198[:7]:
    print(s[:650])

print("\n=== SAVED FORWARD FILES ===")
for j in range(1, 17):
    p = Path(f"/work/fwd_{j}")
    fs = sorted(p.glob("*"))
    print(j, [x.name for x in fs])
    for x in fs:
        if x.name.startswith("dose_edit_") and x.suffix == ".csv":
            d199 = pd.read_csv(x)
            if "edited" in x.name or j <= 2:
                print(" ", x.name, "rows",len(d199),
                      "columns",list(d199.columns),
                      "first",d199.head(1).to_dict("records"))
print("\n=== BRANCH FUNCTION SIGNATURES ===")
import inspect
for fn in (branch195, gelu195, q196):
    print(fn.__name__, inspect.signature(fn))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 199\nRecovering carrier cache and actual saved forward summaries. Six-lag record-derived response kernels suggest substantial insulin and NE effects; large hidden-slot changes alone are not sufficient validation. No edits installed. Counts: 199 Python runs, 16 forwards.\n")
```