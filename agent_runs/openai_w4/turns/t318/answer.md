The architecture confirms that the edit changes how recorded actions enter the recurrent state, not the rollout procedure. I’ll test the symmetric norepinephrine candidate with gain 1.0; its CPU NLL cost was nearly unchanged. GPU rollouts will show whether symmetry improves the weak doubling response and excessive halving response.

```python
import dis
import numpy as np
import pandas as pd
import torch
from pathlib import Path

print("NE CONSTRUCTION")
dis.dis(symmetric_ne271)
dis.dis(symmetric_ne272)

backup275 = {n: torch.as_tensor(load(n)).clone() for n in edited()}
candidate275 = r273_candidates[(1.0, True)]
changes275 = {}
for n, v in candidate275.items():
    t = torch.as_tensor(v, dtype=torch.float32)
    assert tuple(t.shape) == tuple(SHAPES[n]), (n, t.shape, SHAPES[n])
    old = torch.as_tensor(load(n), dtype=torch.float32)
    if not torch.equal(t, old):
        changes275[n] = float(torch.linalg.vector_norm(t - old))
        save_edit(n, t.clone())
print("SAVED CHANGES", changes275)

paths275 = sorted(
    Path("/work").glob("fwd_*/factual_edited.csv"),
    key=lambda p: int(p.parent.name.split("_")[-1])
)
print("LATEST FACTUAL FILE", paths275[-1])
fact275 = pd.read_csv(paths275[-1])
print("COLUMNS", fact275.columns.tolist())
idcol275 = next(c for c in ("stay_id", "stay") if c in fact275.columns)
validation_ids275 = fact275[idcol275].drop_duplicates().tolist()
print("VALIDATION STAYS", validation_ids275)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 275\n"
            "Provisionally saved shared-gate symmetric NE gain 1.0 with exact opposing "
            "output columns, retaining insulin gain 1.5/gate 26. "
            "CPU screen dNLL +.00369608. Changes: " + repr(changes275) +
            ". Previous submission backed up in backup275. "
            "GPU rollout and factual validation pending.\n")
```