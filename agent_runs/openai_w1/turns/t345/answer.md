The new projection preserved factual fit but produced only modest shifts: −2.482 and +1.130 for insulin, +0.219 and −0.431 for norepinephrine. These remain far below the conditional record-derived estimates. I’ll combine the previously stronger insulin projection with the conservative norepinephrine projection, then test both rollout response and aggregate factual fit. Stronger effects without exceeding the factual-loss allowance would support keeping it.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

def tensor296(x):
    return torch.as_tensor(x, dtype=torch.float32).detach().cpu().clone()

candidate295 = {n: tensor296(load(n)) for n in edited()}

P296 = torch.stack([
    tensor296(fits286[("feedback4", 10.0)])[:, 0],
    tensor296(fits286[("feedback4", 100.0)])[:, 1],
], dim=1)
gains296 = torch.tensor([-4.0, 2.0], dtype=torch.float32)
delta296 = tensor296(B286) @ torch.diag(gains296) @ P296.T
original296 = tensor296(load_original("core.b_in.weight"))
assert delta296.shape == original296.shape
assert torch.isfinite(delta296).all()
save_edit("core.b_in.weight", original296 + delta296)
candidate296 = {n: tensor296(load(n)) for n in edited()}

print("Installed candidate296: insulin projection penalty10, gain -4; NE penalty100, gain 2")
print("Injection norm:", float(torch.linalg.vector_norm(delta296)))

for k in [32, 34, 35, 36]:
    folder = Path(f"/work/fwd_{k}")
    a, b = folder / "factual_original.csv", folder / "factual_edited.csv"
    if a.exists() and b.exists():
        da, db = pd.read_csv(a), pd.read_csv(b)
        cols = [c for c in da.columns if "nll" in c.lower() and c in db.columns]
        print("Factual", k, "rows", len(da), "NLL columns", cols)
        for c in cols:
            print(c, "original", float(da[c].mean()),
                  "edited", float(db[c].mean()),
                  "delta", float(db[c].mean() - da[c].mean()))

text296 = Path("/work/notes.md").read_text()
print("\nEARLY RECORD-IDENTIFICATION NOTES\n", text296[:7000])
pos296 = text296.find("## Run 236")
if pos296 >= 0:
    print("\nBASELINE EDIT NOTES\n", text296[pos296:pos296 + 1800])

header296 = pd.read_csv("/data/tuning.csv", nrows=1)
idcol296 = next((c for c in header296.columns
                if c.lower() in ["stay_id", "stay", "icustay_id"]), None)
print("Tuning columns:", list(header296.columns))
if idcol296 is not None:
    all_tuning_ids296 = sorted(
        int(v) for v in pd.read_csv("/data/tuning.csv", usecols=[idcol296])[idcol296].unique()
    )
    tuning_batches296 = [all_tuning_ids296[i:i+100]
                         for i in range(0, len(all_tuning_ids296), 100)]
    print("Tuning ID count/range:", len(all_tuning_ids296),
          all_tuning_ids296[0], all_tuning_ids296[-1])
    print("Consecutive:", all_tuning_ids296 ==
          list(range(all_tuning_ids296[0], all_tuning_ids296[-1] + 1)))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 296\n"
            "Forward36 candidate295 independent factual delta +.00155 passes the local gate. "
            "Forward37 matched rollout shifts [-2.482,+1.130,+.219,-.431], still much smaller "
            "than conditional six-lag record estimates. Saved candidate295 in kernel. "
            "Installed mixed candidate296: feedback4 penalty10 insulin column gain -4, "
            "penalty100 NE column gain 2, retaining safe236 edits. "
            "This is unvalidated; evaluate aggregate factual fit and rollout response. "
            "Counts: 296 Python runs, 37 forwards.\n")
```