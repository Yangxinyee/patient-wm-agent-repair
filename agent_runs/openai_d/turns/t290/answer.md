The reconstruction matches exactly. Increasing norepinephrine strength to 0.70 leaves insulin unchanged and costs about 0.004 on training subsets, but 0.00631 on the tested tuning cohort. I’ll install it only as a reversible trial and assess factual fit across all 400 tuning stays. A pooled increase comfortably below 0.005 would justify testing its stronger rollout response; otherwise I’ll back off.

```python
import numpy as np
import torch
from pathlib import Path

np.savez_compressed(
    "/work/cancellation50_f33.npz",
    **{k: v.numpy() for k, v in r244_current.items()}
)

r247_trial = r246_candidates[(.94, .70)]
for k, v in r247_trial.items():
    save_edit(k, torch.from_numpy(v.copy()))

print("Installed trial: insulin amplitude .94; NE amplitude .70; cancellation 1.0.")
print("Changed tensors:", edited())
print("Cached factual deltas:",
      r246_table[(r246_table.ins_amp == .94) &
                 (r246_table.ne_amp == .70)].to_dict("records"))

notes = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)(?=^## )", notes)
s5 = next((s for s in sections if re.match(r"## Run 5\b", s)), "")
print("\n=== LONG-HORIZON IDENTIFICATION SUMMARY ===")
print(s5[-2200:])
for run in [6, 7]:
    s = next((s for s in sections if re.match(fr"## Run {run}\b", s)), "")
    print(s[:900])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 247: reversible stronger-NE population-budget trial\n"
        "Installed insulin .94 / NE .70 with training-only nonlinear cancellation. "
        "Cached deltas: train fit .004418, train-held-out .003792, tuning100–199 "
        ".006311. Not accepted yet: will evaluate the full 400-stay tuning mixture "
        "and retain only with adequate pooled factual margin. Saved validated "
        ".50 checkpoint to /work/cancellation50_f33.npz and r244_current.\n"
    )
```
