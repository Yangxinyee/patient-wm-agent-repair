The script stopped because one audit-file path was wrong; **no weights changed**. I’ll locate the actual factual files and stage the same precomputed norepinephrine-focused candidate. The acceptance test is unchanged: better dose responses without exceeding the factual-loss limit.

```python
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import re

# Preserve the audited fallback and stage the already-fitted alternative first.
backup296 = {n: torch.as_tensor(load(n)).clone() for n in edited()}
Tbase296 = grid286[(.35, .3, 8)]["T"]
assert np.max(np.abs(torch.as_tensor(load("attr_heads.value_bin.2.weight")).numpy().T
                     - Tbase296[:192])) < 1e-6
candidate296_key = (.35, .5, 4.0)
T296 = grid286[candidate296_key]["T"]
assert T296.shape == (193, 17) and np.isfinite(T296).all()
save_edit("attr_heads.value_bin.2.weight", torch.tensor(T296[:192].T.copy(), dtype=torch.float32))
save_edit("attr_heads.value_bin.2.bias", torch.tensor(T296[192].copy(), dtype=torch.float32))
print("STAGED:", candidate296_key, grid286[candidate296_key]["score"])

# Find the latest pre-strong factual audit for each missing fallback cohort.
cohort_sets296 = [set(tuning_ids294[i:i+100]) for i in range(0, 400, 100)]
fallback_files296 = {0: Path("/work/fwd_35"), 2: Path("/work/fwd_36")}
for p296 in sorted(Path("/work").glob("fwd_*/factual_edited.csv"),
                   key=lambda p: int(p.parent.name.split("_")[1])):
    number296 = int(p296.parent.name.split("_")[1])
    if number296 >= 34:
        continue
    d296 = pd.read_csv(p296)
    if "stay_id" not in d296.columns:
        continue
    stays296 = set(d296["stay_id"])
    for j296 in [1, 3]:
        if stays296 == cohort_sets296[j296]:
            fallback_files296[j296] = p296.parent

fallback_audit296 = {}
groups296 = []
for j296, p296 in sorted(fallback_files296.items()):
    o296 = pd.read_csv(p296 / "factual_original.csv")
    e296 = pd.read_csv(p296 / "factual_edited.csv")
    cols296 = [c for c in o296.columns if "nll" in c.lower()]
    col296 = "nll" if "nll" in cols296 else cols296[0]
    assert len(o296) == len(e296)
    delta296 = float(e296[col296].mean() - o296[col296].mean())
    fallback_audit296[j296] = (len(o296), delta296, p296.name)
    print("FALLBACK COHORT", j296, fallback_audit296[j296])
    if "stay_id" in o296.columns:
        og296 = o296.groupby("stay_id")[col296].agg(["sum", "count"])
        eg296 = e296.groupby("stay_id")[col296].agg(["sum", "count"])
        groups296.append(pd.DataFrame({"dsum": eg296["sum"]-og296["sum"],
                                      "n": og296["count"]}))
if len(fallback_audit296) == 4:
    total296 = sum(v[0] for v in fallback_audit296.values())
    pooled296 = sum(v[0]*v[1] for v in fallback_audit296.values()) / total296
    print("FALLBACK POOLED DELTA:", pooled296, "n=", total296)
    if len(groups296) == 4:
        g296 = pd.concat(groups296)
        r296 = g296["dsum"] - pooled296*g296["n"]
        se296 = np.sqrt(len(g296)/(len(g296)-1)*np.sum(r296**2))/g296["n"].sum()
        print("Cluster SE:", se296, "95% one-sided upper:", pooled296+1.645*se296)

print("=== LATER TARGET NOTES ===")
text296 = Path("/work/notes.md").read_text()
for s296 in text296.split("\n## "):
    title296 = s296.splitlines()[0] if s296 else ""
    if re.search(r"\b(83|97)\b", title296):
        print(s296[:1100])
for name296 in ["cf_targets83", "alpha_targets97"]:
    for key296 in [("insulin", 2.0), ("norepinephrine", 2.0)]:
        v296 = globals()[name296][key296]
        if isinstance(v296, dict):
            desc296 = {
                k: (float(x) if isinstance(x, (int, float, np.number)) else
                    {"type": type(x).__name__, "shape": getattr(x, "shape", None)})
                for k, x in v296.items()
            }
        else:
            desc296 = {"type": type(v296).__name__, "shape": getattr(v296, "shape", None)}
        print(name296, key296, desc296)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 296 / NE-focused alternative staged\n"
        "Run295 stopped on a missing artifact path before changing any weights. "
        "Preserved audited fallback in backup296. Staged grid286(.35,.5,4), "
        "changing only final value-head weights/bias; GPU audit pending. "
        f"Recovered fallback factual files/results: {fallback_audit296}.\n"
    )
```