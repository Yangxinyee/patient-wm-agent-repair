Probability weighting yields only a negligible overall gain, so it does not establish useful factual headroom. I’ll reject these head changes and restore the better-validated insulin baseline. Next I’ll review the measured recurrent-edit frontier to see whether a small norepinephrine repair can fit the remaining NLL budget.

```python
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch

# Reject trial219 and return to the previously validated insulin baseline.
dbase231 = np.load("/work/baseline77.npz", allow_pickle=False)["delta"]
save_edit(
    "core.b_in.weight",
    torch.as_tensor(load_original("core.b_in.weight"), dtype=torch.float32)
    + torch.as_tensor(dbase231, dtype=torch.float32)
)
print("Restored baseline77. Current edits:", edited())

print("\nMeasured paired factual frontier:")
frontier231 = []
for k in range(1, 27):
    po = Path(f"/work/fwd_{k}/factual_original.csv")
    pe = Path(f"/work/fwd_{k}/factual_edited.csv")
    if not (po.exists() and pe.exists()):
        continue
    o, e = pd.read_csv(po), pd.read_csv(pe)
    common = [c for c in o.columns if c in e.columns]
    nc = next((c for c in ["nll", "value_nll", "fine_bin_nll"]
               if c in common), None)
    if nc is None:
        print("Unrecognized schema", k, list(o.columns))
        continue
    sc = next((c for c in ["stay_id", "stay"] if c in common), None)
    frontier231.append({
        "forward": k, "readings": len(o),
        "stays": o[sc].nunique() if sc else None,
        "first_stay": int(o[sc].min()) if sc else None,
        "last_stay": int(o[sc].max()) if sc else None,
        "original": float(o[nc].mean()),
        "edited": float(e[nc].mean()),
        "cost": float(e[nc].mean()-o[nc].mean())
    })
print(pd.DataFrame(frontier231).to_string(index=False))

print("\nSaved candidate schemas:")
for filename in [
    "insulin_candidate63.npz", "insulin_trial219.npz",
    "ne_candidate148.npz", "ne_candidate151.npz",
    "ne_candidate154.npz", "ne_candidate155.npz",
    "ne_candidate199.npz", "ne_candidate82.npz"
]:
    with np.load("/work/"+filename, allow_pickle=False) as a:
        print(filename, [(k, a[k].shape) for k in a.files])
        if "delta" in a.files:
            d = a["delta"].astype(np.float64)
            print("  delta norm:", float(np.linalg.norm(d)),
                  "baseline projection:",
                  float(np.sum(d*dbase231)/np.sum(dbase231.astype(np.float64)**2)))

print("\nEarlier NE frontier evidence:")
text231 = Path("/work/notes.md").read_text()
lines231 = text231.splitlines()
selected231 = []
for i, line in enumerate(lines231):
    if any(s in line.lower() for s in [
        "ne_candidate148", "ne_candidate151", "ne_candidate154",
        "ne_candidate155", "ne_candidate199", ".00652",
        "forward23", "forward 23", "run 199"
    ]):
        selected231.append("\n".join(lines231[max(0,i-1):min(len(lines231),i+3)]))
print("\n---\n".join(selected231[:14])[:6500])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 231: reject negligible head gain; restore baseline77\n"
        "Probability-weighted head regression offered no material overall "
        "factual gain, so no head edit is accepted. Rejected trial219 after "
        "fresh-sample factual cost .00567 and restored baseline77 exactly. "
        "Auditing measured NE frontier for a conservative add-on.\n"
        "Python runs: 231; forwards: 26.\n"
    )
```