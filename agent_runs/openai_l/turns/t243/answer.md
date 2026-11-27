The two-unit candidate has a validation NLL increase of 0.00178 and stronger immediate dose responses than the current edit, though it still falls short of the targets. I’ll stage it with a rollback copy and test the actual model. A factual increase above 0.005 would reject it regardless of its causal benefit.

```python
import numpy as np, torch
from pathlib import Path

backup219 = {n: np.asarray(load(n)).copy() for n in edited()}
candidate219 = candidates215[("linear2", 1.0)]
arch219 = candidate219["arch"]
print("Value-head tensor shapes:",
      {n: SHAPES[n] for n in NAMES if n.startswith("attr_heads.value_bin.")})

W1219 = np.asarray(load_original("attr_heads.value_bin.0.weight")).copy()
b1219 = np.asarray(load_original("attr_heads.value_bin.0.bias")).copy()
W1219[arch219["units"]] = arch219["rows"][:, :192]
b1219[arch219["units"]] = arch219["rows"][:, 192]

check219 = np.asarray(Xraw204)[:, :20]
if check219.shape[-1] == 193:
    pre219 = check219[..., :192] @ W1219.T + b1219
else:
    pre219 = check219 @ W1219.T + b1219
Fcheck219 = np.concatenate(
    [gelu214(pre219), np.ones(pre219.shape[:-1] + (1,))], axis=-1)
err219 = float(np.max(np.abs(Fcheck219 - arch219["F"][:, :20])))
print("Staged architecture reconstruction max error:", err219)
assert err219 < 0.002, "Feature mismatch: no edits saved."

save_edit("attr_heads.value_bin.0.weight", torch.from_numpy(W1219.astype(np.float32)))
save_edit("attr_heads.value_bin.0.bias", torch.from_numpy(b1219.astype(np.float32)))
save_edit("attr_heads.value_bin.2.weight",
          torch.from_numpy(candidate219["T"][:192].T.astype(np.float32)))
save_edit("attr_heads.value_bin.2.bias",
          torch.from_numpy(candidate219["T"][192].astype(np.float32)))

print("Staged linear2 eta=1:", candidate219["train_delta"],
      candidate219["val_delta"], candidate219["slopes"])
print("Current edits:", edited())
print("Prior factual audit stay IDs:",
      sorted(pd.read_csv("/work/fwd_12/factual_original.csv")["stay_id"].unique().tolist())
      if "stay_id" in pd.read_csv("/work/fwd_12/factual_original.csv").columns
      else pd.read_csv("/work/fwd_12/factual_original.csv").columns.tolist())

with Path("/work/notes.md").open("a") as f219:
    f219.write("\n## Run 219 / nonlinear head candidate staged\n"
               "Staged linear2 eta=1: replaced two first-layer GELU units and fitted "
               "the final value layer analytically. Offline train/validation NLL "
               "deltas +.00481/+.00178; x2 immediate slopes I -1.712, NE +1.465. "
               "All prior edits retained in backup219 for rollback. "
               "Pending paired GPU factual audit; no causal success claimed.\n")
```
