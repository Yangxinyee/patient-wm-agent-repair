These attention scores confirm that the latent token is effectively ignored on drug bundles. I’ll install an exact diagnostic probe before changing the action pathway. If its observation and action coordinates match the local encodings, the reconstruction error is in context; mismatches will instead pinpoint an encoding problem.

```python
import numpy as np
import torch
import json
from pathlib import Path

r67_probe_columns = np.concatenate([
    np.linspace(0, 191, 8, dtype=int),
    192 + np.linspace(0, 191, 16, dtype=int),
    384 + np.linspace(0, 191, 8, dtype=int)
])
r67_probe_scale = 0.05
r67_probe_bias = np.median(compact_array(sx[:, 960:992]), axis=0).astype(np.float32)

r67_W0 = np.zeros(SHAPES["z_post.0.weight"], dtype=np.float32)
r67_b0 = np.zeros(SHAPES["z_post.0.bias"], dtype=np.float32)
r67_W2 = np.zeros(SHAPES["z_post.2.weight"], dtype=np.float32)
r67_b2 = compact_array(load_original("z_post.2.bias")).copy()
for j, column in enumerate(r67_probe_columns):
    r67_W0[j, column] = 1.0
    r67_W0[32 + j, column] = -1.0
    r67_W2[j, j] = r67_probe_scale
    r67_W2[j, 32 + j] = -r67_probe_scale
r67_b2[:32] = r67_probe_bias

for name, array in [
    ("z_post.0.weight", r67_W0), ("z_post.0.bias", r67_b0),
    ("z_post.2.weight", r67_W2), ("z_post.2.bias", r67_b2)
]:
    save_edit(name, torch.tensor(array))

with np.load("/work/fwd_3/states_original.npz") as r67_npz:
    r67_probe_stays = np.unique(r67_npz["stay_id"]).astype(int).tolist()

print("Temporary probe installed:", edited())
print("Columns:", r67_probe_columns.tolist())
print(json.dumps({"type": "states", "model": "edited", "split": "train",
                  "stays": r67_probe_stays, "bundles": "all"}))

with Path("/work/notes.md").open("a") as f:
    f.write("\nDiagnostic probe: exact GELU pairs expose selected context, observation "
            "and action coordinates in posterior means. Temporary four-tensor edit; "
            "restore immediately after acquiring probe states. Prelude latent attention "
            "on drug bundles is <1e-5, explaining the ineffective posterior pilot.\n")
```
