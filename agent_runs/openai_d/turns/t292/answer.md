The stronger trial increased NLL by 0.00920 on the first cohort, contradicting the hoped-for factual margin. I’ll reject it and restore the validated 0.50 trial. Next I’ll inspect the existing caches and earlier factual checks to determine whether broader training-only cancellation can help; it must generalize beyond the fitting stays to be useful.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import inspect

for k, v in r244_current.items():
    save_edit(k, v.clone())
print("Restored NE .50 cancellation checkpoint.")

print("\n=== FORWARD FACTUAL HISTORY ===")
for k in range(1, 35):
    p = Path(f"/work/fwd_{k}/factual_original.csv")
    q = Path(f"/work/fwd_{k}/factual_edited.csv")
    if not p.exists() or not q.exists():
        continue
    a, b = pd.read_csv(p), pd.read_csv(q)
    if k == 34:
        print("Columns:", list(a.columns))
    nc = next((c for c in a.columns if c.lower() in
               ("nll", "value_nll", "fine_nll", "value_bin_nll")), None)
    sc = next((c for c in a.columns if "stay" in c.lower()), None)
    if nc:
        print(k, "n:", len(a),
              "first stay:", a[sc].iloc[0] if sc else None,
              "delta:", round(float(b[nc].mean() - a[nc].mean()), 7))
    else:
        print(k, "columns:", list(a.columns))

print("\n=== STATE FILES ===")
for p in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    print(str(p))

print("\n=== CACHE/STATE HELPER FUNCTIONS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        if any(w in name.lower() for w in
               ("cache", "state", "head", "input", "view", "bundle", "replay",
                "tensor", "feature", "attr", "calib")):
            print(name, list(inspect.signature(obj).parameters))
            if "cache" in name.lower():
                print("  globals:", obj.__code__.co_names)

print("\n=== CACHE INVENTORY ===")
for name in ["cache", "cc", "c210_train", "c210_tune"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, {k: tuple(v.shape) if hasattr(v, "shape")
                     else type(v).__name__ for k, v in obj.items()})
        if isinstance(obj.get("meta"), pd.DataFrame):
            print("  meta:", list(obj["meta"].columns))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Forward 34 / Run 248: stronger-NE trial rejected\n"
        "NE .70 cancellation increased first-cohort tuning factual NLL by .00920 "
        "(1.64381 vs 1.63461), undermining population-budget hypothesis. Restored "
        "validated .50 tensors from r244_current. Need broader factual validation "
        "and/or training-only cancellation that generalizes; no stronger edit accepted.\n"
    )
```
