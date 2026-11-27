The architecture confirms that only value predictions are edited directly; the other heads retain their original weights. The audit output was truncated before the final causal estimates and edit helpers. I’ll recover those compactly, plus the remaining validation IDs. Stable estimates across methods would support the targets; disagreement would call for caution.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("=== FINAL RECORD-BASED TARGETS ===")
for fn in ["kernel_targets.csv", "sensor_iv.csv", "dynamic_iv.csv"]:
    df = pd.read_csv("/work/" + fn)
    print(fn, df.shape)
    print(df.head(8).to_string(index=False))

print("\n=== EDIT HELPERS ===")
for k, v in sorted(list(globals().items())):
    if not k.startswith("_") and inspect.isfunction(v) and v.__module__ == "__main__":
        try:
            print(k, str(inspect.signature(v)))
        except Exception:
            print(k)

print("\n=== SELECTED IDENTIFICATION NOTES ===")
sections249 = Path("/work/notes.md").read_text().split("\n## ")
chosen249 = [s for s in sections249
             if any(t in s.split("\n")[0].lower()
                    for t in ["identif", "kernel", "instrument", "causal", "iv ", "iv:", "sensor"])]
print("\n".join(s[:650] for s in chosen249[-6:]))

print("\n=== CURRENT CHECKPOINT ===")
with np.load("/work/safer_candidate247.npz") as ck249:
    print([(k, ck249[k].shape) for k in ck249.files])

print("\n=== COMPACT LIVE-OBJECT INDEX ===")
for k, v in sorted(list(globals().items())):
    if k.startswith("_") or not any(t in k.lower() for t in
        ["coef", "calib", "cache", "target", "frontier", "ids", "batch"]):
        continue
    if isinstance(v, np.ndarray):
        print(k, v.shape)
    elif isinstance(v, dict):
        print(k, "keys", list(v)[:6])
    elif isinstance(v, pd.DataFrame):
        print(k, "DataFrame", v.shape)
    elif isinstance(v, (list, tuple)):
        print(k, "length", len(v))

tuning_ids249 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique()
print("\n=== REMAINING TUNING BATCH, IDS 300–399 ===")
print([int(x) for x in tuning_ids249[300:400]])

with Path("/work/notes.md").open("a") as fh249:
    fh249.write(
        "\n## Run 249: compact recovery\n"
        "- Architecture read in run 248; its output was truncated before final targets/helpers.\n"
        "- No weights changed. Recovering final causal targets and the last independent tuning batch.\n"
    )
```