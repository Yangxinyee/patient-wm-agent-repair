Regularization reduced the training-replay factual cost: the ridge-10 candidate increased NLL by about 0.00345, versus 0.00447 for the current edit. I’ll provisionally save it and test held-out rollouts. It is useful only if that extra factual margin accompanies better dose responses, rather than merely weaker responses.

```python
from pathlib import Path
import numpy as np, pandas as pd, torch

backup_before210 = {n: tensor32_209(load(n)) for n in edited()}
selected210 = candidates209["ridge10_scale-1"]
assert selected210["core.b_in.weight"].dtype == torch.float32
save_edit("core.b_in.weight", selected210["core.b_in.weight"])

print("Selected CPU dNLL:",
      results209["ridge10_scale-1"]["nll"] - results209["original"]["nll"])
print("Changed only core.b_in.weight; all other saved tensors retained.")

last_rollout210 = pd.read_csv("/work/fwd_19/dose_edit_edited.csv")
print("ROLLOUT COLUMNS", list(last_rollout210.columns))
print(last_rollout210.head(3).to_string(index=False))
stay_column210 = next(c for c in ["stay_id", "stay"] if c in last_rollout210.columns)
ids210 = sorted(int(x) for x in last_rollout210[stay_column210].unique())
assert len(ids210) <= 100
print("MATCHED TUNING STAYS", ids210)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 210\n"
        "Float32 rerun succeeded. Original training-replay NLL 1.635536; prior submitted "
        "1.640008 (delta .004472). Ridge-10 NE direction at connection scale -1 yielded "
        "1.638988 (delta .003451). Provisionally saved that B_in only, preserving all "
        "channel and insulin tensors. Previous submitted edits backed up in backup_before210. "
        "Held-out factual and paired dose rollouts required before acceptance; response orientation "
        "and magnitude are not established by this factual screen.\n"
    )
```