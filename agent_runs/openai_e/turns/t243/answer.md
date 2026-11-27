The recovered estimates put the working targets near −13.6/+6.8 for insulin and +4.2 to +4.6/−2.1 to −2.3 for norepinephrine. The current repair is therefore closer on insulin doubling than on the other edits. I’ll extract the measured validation table and saved constructions without the lengthy history. That will show whether any existing candidate offers a better feasible trade-off.

```python
from pathlib import Path
import numpy as np, pandas as pd, re

out211 = []
out211.append("=== FORWARD SCHEMAS ===")
for n in (1, 10, 14, 15, 17, 18, 19, 20, 21, 23):
    p = Path(f"/work/fwd_{n}")
    out211.append(f"{n}: {[x.name for x in p.iterdir()]}")
    if n in (1, 21, 23):
        for x in sorted(p.glob("*.csv")):
            d = pd.read_csv(x)
            out211.append(f"{x.name}: {d.shape}, {d.columns.tolist()}")
            out211.append(str(d.head(1).to_dict("records")))

out211.append("\n=== ARCHIVE SCHEMAS ===")
for fn in ("ne_reduced107.npz", "ne_trial103.npz", "ne_candidate151.npz",
           "ne_candidate154.npz", "ne_candidate155.npz", "ne_candidate199.npz",
           "insulin_candidate58.npz", "insulin_candidate63.npz"):
    with np.load("/work/" + fn) as z:
        out211.append(f"{fn}: " + str({k: z[k].shape for k in z.files}))

out211.append("\n=== INSULIN CONSTRUCTION SUMMARIES ===")
for s in re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M):
    h = s.splitlines()[0] if s.splitlines() else ""
    if re.search(r"(Run[s ]*(?:5[5-9]|6[0-8])\b|Forward 10\b)", h):
        out211.append(s[:420])

out211.append("\n=== HELPER NAMES ===")
out211.append(str([k for k,v in list(globals().items())
                  if callable(v) and getattr(v, "__module__", "") == "__main__"]))
print("\n".join(out211)[:8800])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 211: strictly bounded artifact recovery\n"
            "Working record-based targets: insulin about -13.6/+6.8; NE about +4.2..4.6/-2.1..2.3 "
            "on early tuning reached hours. Recovering saved edit schemas and measured forward "
            "outputs with a hard output bound. No weights changed. Python runs: 211; forwards: 23.\n")
```
