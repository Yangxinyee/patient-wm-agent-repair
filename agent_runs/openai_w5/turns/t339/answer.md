The 40% strength trial raises factual NLL by 0.00832 nats on the fresh block, so I’m rejecting it. Most deterioration is in glucose, consistent with the increased readout strength being too costly. I’ll restore the unscaled 1,250-second candidate, then test it on the same stays to separate strength cost from persistence cost.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

save_edit(
    "attr_heads.value_bin.2.weight",
    BASE289["attr_heads.value_bin.2.weight"].clone()
)

original290 = pd.read_csv("/work/fwd_35/factual_original.csv")
trial290 = pd.read_csv("/work/fwd_35/factual_edited.csv")
keys290 = ["stay_id", "hour", "event", "observed"]
assert original290[keys290].equals(trial290[keys290])
audit290 = original290[keys290].copy()
audit290["delta_nll"] = trial290["nll"].to_numpy() - original290["nll"].to_numpy()
print("Rejected trial NLL delta:", float(audit290["delta_nll"].mean()))
print(audit290.groupby("event")["delta_nll"].agg(["mean", "count"]).to_string())

clusters290 = audit290.groupby("stay_id")["delta_nll"].agg(["sum", "count"])
rng290 = np.random.default_rng(290)
draws290 = rng290.integers(0, len(clusters290), size=(2000, len(clusters290)))
sums290 = clusters290["sum"].to_numpy()[draws290].sum(axis=1)
counts290 = clusters290["count"].to_numpy()[draws290].sum(axis=1)
print("Stay-bootstrap interval:", np.quantile(sums290 / counts290, [0.025, 0.975]))
print("Restored tau=1250, glucose strength=1.0; all other tensors unchanged.")

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Forward 34 / run 290: reject glucose-strength 1.40\n"
        "- Fresh tuning B factual NLL: 1.62730 vs 1.61898, "
        "delta +0.00832. Fails the local guardrail.\n"
        "- Reverted only the value-head output weights to BASE289 "
        "(tau=1250, glucose strength=1.0).\n"
        "- Next paired factual test on the same fresh stays distinguishes "
        "strength cost from persistence/centering generalization failure.\n"
    )
```