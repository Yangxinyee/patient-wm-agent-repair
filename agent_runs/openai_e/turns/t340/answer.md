Candidate A met the factual-loss limit on one cohort, but still underestimated the drug effects. Candidate B is installed but untested. I’ll recover the architecture and saved edit definitions before validating B across all 400 tuning stays. Stronger dose responses with factual loss comfortably below the limit would support keeping it.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nDATA README")
print(Path("/data/README.md").read_text())

notes = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-BASED IDENTIFICATION NOTES")
print(notes[:6500])

for filename in ["joint236.npz", "joint274.npz", "joint283.npz", "joint284.npz"]:
    with np.load("/work/" + filename, allow_pickle=True) as z:
        print("\nARCHIVE", filename)
        for key in z.files:
            a = z[key]
            if a.size <= 12:
                print(key, a)
            else:
                print(key, "shape", a.shape, "dtype", a.dtype)

print("\nTUNING SCHEMA")
preview = pd.read_csv("/data/tuning.csv", nrows=2)
print(preview.to_string(index=False))
print("\nCURRENT EDITS", edited())
print("\nRECENT FORWARD FILES")
for k in [36, 37, 38, 39, 40, 41, 42]:
    p = Path(f"/work/fwd_{k}")
    print(k, [x.name for x in sorted(p.iterdir())] if p.exists() else "missing")

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 285: restart audit before population validation\n"
        "Candidate B remains installed and unaccepted. Recovered architecture, "
        "record-identification notes, archive contracts, and tuning schema. "
        "No weights changed. Next: paired factual validation across all400 tuning stays "
        "and exact dose rollout; retain only if population fit supports the constraint.\n"
        "Python runs: 285; forwards: 42.\n"
    )
```