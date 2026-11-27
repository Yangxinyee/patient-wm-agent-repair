The causal estimates are stable across dose and time subsets: approximately −1.99 glucose units per insulin-rate unit and +3.29 MAP units per norepinephrine-rate unit. The unresolved obstacle is the head reconstruction. I’ll temporarily remove the sampled latent contribution from the reading view, leaving the factual recurrent states unchanged. This diagnostic should let the local head match GPU predictions nearly exactly; a mismatch would refute the reconstruction.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

assert edited() == [], "Unexpected existing edits; stop before the diagnostic."

probe83_factual = pd.read_csv("/work/fwd_4/factual_original.csv")
probe83_staycol = next(c for c in ("stay_id", "stay") if c in probe83_factual.columns)
probe83_stays = sorted(int(x) for x in probe83_factual[probe83_staycol].unique())
assert len(probe83_stays) <= 100

print("Factual columns:", list(probe83_factual.columns))
print("Factual examples:")
print(probe83_factual.head(4).to_string(index=False))
print("Probe stays:", probe83_stays)

with np.load("/work/fwd_3/states_original.npz") as probe83_npz:
    print("State arrays:", {k: probe83_npz[k].shape for k in probe83_npz.files})

probe83_zhead_original = load_original("z_head.weight").clone()
save_edit("z_head.weight", torch.zeros_like(probe83_zhead_original))
print("Diagnostic edits:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 83: deterministic reading-head diagnostic\n"
        "All previous edits were absent. Temporarily zeroed z_head.weight only, "
        "to remove prior-draw uncertainty from factual reading predictions on "
        "the 100 training stays with saved original states. This is a diagnostic, "
        "not a proposed final repair. Require near-exact local/GPU head agreement "
        "before installing another gated correction; restore z_head afterward.\n"
    )
```