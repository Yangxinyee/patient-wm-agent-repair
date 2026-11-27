The architecture confirms that the edit affects recurrent state updates, so factual preservation needs direct testing. The audit was truncated before the saved edits and stay IDs. I’ll retrieve those compactly and verify that the installed weights match candidate B before starting paired validation.

```python
import numpy as np
import pandas as pd
from pathlib import Path

def audit_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

current_B = audit_numpy(load("core.b_in.weight"))
original_B = audit_numpy(load_original("core.b_in.weight"))
print("CURRENT", current_B.shape, "delta_norm",
      float(np.linalg.norm(current_B - original_B)), "edits", edited())

for filename in ["joint236.npz", "joint274.npz", "joint284.npz"]:
    with np.load("/work/" + filename, allow_pickle=True) as z:
        print("\nARCHIVE", filename)
        for key in z.files:
            a = z[key]
            if a.size <= 12:
                print(key, str(a)[:250])
            else:
                print(key, a.shape, a.dtype)
            if a.shape == current_B.shape and np.issubdtype(a.dtype, np.number):
                print("  distance_to_installed",
                      float(np.linalg.norm(a.astype(np.float32) - current_B)))

tuning_ids_audit = np.sort(
    pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique()
)
print("\nTUNING IDS", len(tuning_ids_audit),
      "min", tuning_ids_audit.min(), "max", tuning_ids_audit.max(),
      "contiguous", bool(np.all(np.diff(tuning_ids_audit) == 1)))
for start in [0, 100, 200, 300]:
    block = tuning_ids_audit[start:start + 100]
    print("COHORT", start, "first", block[0], "last", block[-1])

for k in [36, 37, 38, 39, 41]:
    p = Path(f"/work/fwd_{k}/factual_original.csv")
    q = Path(f"/work/fwd_{k}/factual_edited.csv")
    if p.exists() and q.exists():
        fo, fe = pd.read_csv(p), pd.read_csv(q)
        print("\nFACTUAL", k, "rows", len(fo), len(fe),
              "columns", list(fo.columns))
        print("original first row", fo.iloc[0].to_dict())
        print("edited first row", fe.iloc[0].to_dict())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 286: compact validation-contract audit\n"
        "Previous output truncated before archives and IDs. Retrieved candidate "
        "tensor contracts, checked installed/archive distances, and recovered "
        "tuning ID ranges and factual schemas. No weights changed.\n"
        "Python runs: 286; forwards: 42.\n"
    )
```