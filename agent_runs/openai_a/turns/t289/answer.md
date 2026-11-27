Lowering the threshold to 0.035 leaves inactive-hour predictions almost unchanged in the cached screen and slightly improves cached tuning NLL. I’ll install that small, reversible bias shift and test the exact rollout. A stronger halving response without losing the doubling response would support the gate-switching hypothesis.

```python
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

gate035_checkpoint = {}
for n, v in validated_gated8_checkpoint.items():
    a = np.asarray(v).copy()
    if n == "attr_heads.value_bin.0.bias":
        a[np.asarray(sc_nodes, dtype=int)] += np.float32(180.0*(0.05-0.035))
    gate035_checkpoint[n] = torch.as_tensor(a, dtype=torch.float32).clone()

for n, v in gate035_checkpoint.items():
    save_edit(n, v)

gate035_config = dict(activity_gate_config)
gate035_config["activity_centre"] = 0.035

_tune_ids248 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].drop_duplicates().tolist()
_tune_first100248 = [int(x) for x in _tune_ids248[:100]]

with Path("/work/notes.md").open("a") as fh:
    fh.write(
        "\n## Gate-threshold trial; Python 248, forwards 31/50\n"
        "- Current tensors matched validated_gated8_checkpoint exactly before editing.\n"
        "- Lowered activity centre from .05 to .035 by adding 2.7 to the two graft hidden biases.\n"
        "- Cached tuning NLL difference versus gated8 is -.000269; inactive-hour change negligible.\n"
        "- This is a reversible, UNACCEPTED trial. Exact GPU rollout and factual validation required.\n"
        "- validated_gated8_checkpoint remains the full400-tested fallback.\n"
    )

print("Bias shift:", 180.0*(0.05-0.035))
print("Current edits:", edited())
print("Next rollout request:")
print(json.dumps({"type": "dose_edit", "model": "both", "split": "tuning",
                  "stays": _tune_first100248}))
```
