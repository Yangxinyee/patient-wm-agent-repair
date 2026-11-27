The candidate was saved, but the artifact lookup failed before validation. I’ll locate the actual forward-output files and recover the stay IDs rather than assume a filename. The factual-fit and rollout checks are still outstanding.

```python
from pathlib import Path
import pandas as pd, re, numpy as np

print("=== RECENT FORWARD ARTIFACTS ===")
for number211 in range(15, 20):
    directory211 = Path(f"/work/fwd_{number211}")
    files211 = sorted(p for p in directory211.rglob("*") if p.is_file())
    print(directory211.name, [str(p.relative_to(directory211)) for p in files211])
    for file211 in files211:
        if file211.suffix == ".csv":
            preview211 = pd.read_csv(file211, nrows=2)
            print(file211.name, "columns:", list(preview211.columns))

print("=== RETAINED TUNING-ID LISTS ===")
for name211 in sorted(list(globals())):
    if not re.search(r"(ids|stays).*(19[6-9]|20[0-6])|(19[6-9]|20[0-6]).*(ids|stays)", name211):
        continue
    value211 = globals()[name211]
    if isinstance(value211, (list, tuple, np.ndarray)):
        array211 = np.asarray(value211)
        if array211.ndim == 1 and len(array211) <= 100:
            print(name211, array211.tolist())

print("B_in matches selected candidate:",
      np.allclose(np.asarray(load("core.b_in.weight")),
                  selected210["core.b_in.weight"].numpy(), atol=1e-6))
with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Runs 210–211\n"
        "Run-210 saved the ridge-10 NE connection at scale -1 (CPU training dNLL .003451), "
        "then failed while looking up a guessed rollout filename; its notebook write did not execute. "
        "Previous saved tensors remain in backup_before210. Locating actual forward artifacts now. "
        "No additional weights changed in Run-211; held-out validation remains outstanding.\n"
    )
```