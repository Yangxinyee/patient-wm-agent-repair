Downward threshold shifts preserved factual fit in the CPU replay. Scaling the whole insulin connection did not, so that matrix also contains compensation we must retain. I’ll test the threshold-26 candidate with the connection unchanged. It will support the gating hypothesis only if rollout asymmetry improves and GPU factual fit still passes.

```python
import numpy as np, torch, pandas as pd
from pathlib import Path

backup_before233 = {n: load(n).detach().clone() for n in edited()}
save_edit("z_post.0.bias", candidates232[(26.0, 1.0)]["z_post.0.bias"])
print("Saved threshold-26 bias only; other tensors unchanged.")

print("\n=== CORRECT ACTION MASKS AND GATE MARGINS ===")
print("Bank keys:", list(repair_bank))
print("indices102:", np.asarray(indices102).shape)
for drug233 in ["insulin", "norepinephrine"]:
    key233 = "has_" + drug233
    if key233 in repair_bank:
        mask233 = np.asarray(repair_bank[key233])[np.asarray(indices102, dtype=int)].astype(bool)
        print(drug233, "bundle count", mask233.sum(),
              "insulin gate quantiles",
              np.round(np.quantile(pre232[mask233].mean(axis=1), [.01,.1,.5,.9,.99]),3))
print("intervention_bank74 references:", intervention_bank74.__code__.co_names)
for factor233 in [0.5, 2.0]:
    obj233 = intervention_bank74("insulin", factor233)
    if isinstance(obj233, dict):
        print("intervention", factor233, "dict", list(obj233))
    elif isinstance(obj233, (list, tuple)):
        print("intervention", factor233, "sequence",
              [(type(v).__name__, tuple(v.shape) if hasattr(v,"shape") else None) for v in obj233])
    else:
        print("intervention", factor233, type(obj233).__name__,
              tuple(obj233.shape) if hasattr(obj233,"shape") else repr(obj233)[:150])

print("\n=== LATEST FORWARD FILES AND STAY IDS ===")
for d233 in [Path("/work/fwd_21"), Path("/work/fwd_20"), Path("/work/fwd_17")]:
    files233 = list(d233.glob("*dose*csv"))
    if files233:
        print(d233.name, [p.name for p in files233])
        frame233 = pd.read_csv(files233[0])
        print("columns", frame233.columns.tolist())
        idcol233 = next((c for c in ["stay_id","stay"] if c in frame233), None)
        if idcol233:
            ids233 = sorted(frame233[idcol233].unique().astype(int).tolist())
            print("stays", ids233)
        break

with open("/work/notes.md", "a") as f233:
    f233.write("\n## Run 233\nSaved only insulin paired-gate bias shift -26 (rows 96/184), leaving B_in and all other tensors at the previously GPU-passing candidate. CPU dNLL .002812. Full insulin-matrix gain scaling failed (~.06 nats), likely because it scales structural compensation too. Backup before this gate edit is backup_before233. Corrected action-mask inspection: active94 is not an insulin mask. GPU factual and rollout validation pending.\n")
```