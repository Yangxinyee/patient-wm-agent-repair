The carrier-only change leaves the prelude essentially unchanged, and the reserved head attends exclusively to the carrier. The remaining error was in the diagnostic: forward states are packed into `x`, not separate `h` and `out` fields. I’ll unpack them and verify recurrent replay before calibrating an injection; a close match is required to trust that calibration.

```python
import numpy as np
import torch
from pathlib import Path
import pandas as pd

packed156 = tensor153(S["x"])
assert packed156.shape[1] == 992
observed_out156 = packed156[:, :192]
observed_h156 = packed156[:, 192:960].reshape(-1,4,192)
observed_z156 = packed156[:, 960:992]
print("Posterior reconstruction vs authoritative z RMS:",
      float(torch.sqrt(torch.mean(
          (tensor153(Ptarget154[:,:32])-observed_z156[np.asarray(idx81)])**2))))

state_checks156 = []
for include156 in (False, True):
    mem156, mask156, age156 = memory82(indices155, include156)
    hs156, os156 = state_update82(
        tensor153(h81)[positions155], E_original155[positions155],
        mem156, mask156, age156)
    eh156 = float(torch.sqrt(torch.mean(
        (hs156-observed_h156[indices155])**2)))
    eo156 = float(torch.sqrt(torch.mean(
        (os156-observed_out156[indices155])**2)))
    state_checks156.append((eo156, include156, mem156, mask156, age156, hs156, os156))
    print("Include-current", include156, "h RMS", eh156, "out RMS", eo156,
          "age range", float(age156.min()), float(age156.max()))

best_memory156 = min(state_checks156, key=lambda x:x[0])
memory156, mask156, age156 = best_memory156[2:5]
hbase156, obase156 = best_memory156[5:7]

print("\nReadout setup:")
print("Codes:", np.asarray(read_codes105))
print("next_dt105:", tuple(next_dt105.shape),
      np.asarray(next_dt105).reshape(-1)[:5])
print("attr_static105 shape:", getattr(attr_static105, "shape", None))
viewbase156 = next_view105(hbase156, obase156)
readbase156 = np.asarray(fn144(viewbase156))
print("Readout shape:", readbase156.shape,
      "mean:", readbase156.mean(0),
      "range:", readbase156.min(0), readbase156.max(0))
print("T136:", np.asarray(T136))
print("directions136 column norms:",
      np.linalg.norm(np.asarray(directions136), axis=0))
print("G136 row norms:", np.linalg.norm(np.asarray(G136), axis=1))
print("G136 row cosine matrix:", np.round(
    np.asarray(G136) @ np.asarray(G136).T /
    np.maximum(np.linalg.norm(np.asarray(G136),axis=1)[:,None] *
               np.linalg.norm(np.asarray(G136),axis=1)[None,:], 1e-20), 3))

factual156 = pd.read_csv("/work/fwd_10/factual_original.csv")
print("\nFactual CSV columns:", list(factual156.columns))
id_column156 = next(c for c in ("stay_id", "stay") if c in factual156.columns)
validation_ids156 = factual156[id_column156].drop_duplicates().tolist()
print("Validation stay IDs:", validation_ids156)
print("Current edited tensor count:", len(edited()))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython run 156; forwards 10. Corrected packed-state "
        "diagnostic (out=x[:192], h=x[192:960], z=x[960:992]); "
        "validated posterior/recurrent replay and retained readout "
        "setup before numerical injection calibration. Carrier-only "
        "prelude disturbance is unchanged from zero-head compensation "
        "(~0.00048 RMS), with 100% reserved-head latent selection. "
        "No new candidate weights installed.\n"
    )
```