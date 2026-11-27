Reversing the NE connection at half strength costs about +0.00447 nats on training replay; full reversal exceeds the limit. I’ll provisionally save the half-strength reversal for GPU validation. It is useful only if it produces positive NE rollout effects and also passes held-out factual fit.

```python
import pandas as pd
from pathlib import Path
import numpy as np
import torch

backup_before203 = {n: torch.as_tensor(load(n)).detach().clone() for n in edited()}
save_edit("core.b_in.weight", base_b202 - 0.5 * unit_ne202)
rW = {n: torch.as_tensor(load(n)).detach().clone() for n in NAMES}

dose_file203 = Path("/work/fwd_17/dose_edit_edited.csv")
dose_table203 = pd.read_csv(dose_file203)
print("DOSE CSV COLUMNS", dose_table203.columns.tolist())
stay_col203 = next(c for c in dose_table203.columns if c in ("stay_id", "stay", "id"))
stays203 = sorted(dose_table203[stay_col203].dropna().unique().tolist())
stays203 = [int(x) if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
            for x in stays203[:100]]
print("STAYS FOR MATCHED GPU TEST", stays203)

original_mean203 = results194[0.5]["mean_nll"] - results194[0.5]["delta_original"]
print("TRAIN REPLAY DNLL",
      {s: float(np.mean(results202[s][1]) - original_mean203)
       for s in results202})
for n in ["preserved_b194", "ne_direction194", "joint194", "joint_decoders193",
          "features193", "target193", "train_b193", "test_b193"]:
    if n in globals():
        x = globals()[n]
        if isinstance(x, dict):
            print(n, {str(k): list(v.shape) if hasattr(v, "shape") else str(type(v))
                      for k, v in x.items()})
        else:
            print(n, list(x.shape) if hasattr(x, "shape") else str(type(x)))

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 203\nProvisionally saved NE B_in scale -.5, with every other channel/insulin weight unchanged. Training replay dNLL is approximately +.00447 versus original; scale -1 fails. Previous passing +.5 candidate saved in backup_before203 and backup_passing200. Matched tuning IDs recovered from fwd_17. GPU factual and rollout validation pending.\n")
```