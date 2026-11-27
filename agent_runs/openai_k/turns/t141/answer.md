The candidate first compresses the original latent token into 30 channels, then adds two correction channels. That means comparing it directly with the original mixes compression and treatment changes. I’ll isolate those two channels and check their factual centering. If they reliably encode separate drug innovations, we can calibrate them independently; otherwise, their construction needs repair.

```python
import numpy as np
import torch
import re
from pathlib import Path

r130_notes = Path("/work/notes.md").read_text()
print("=== DYNAMIC IDENTIFICATION AND TARGETS ===")
r130_start = r130_notes.find("## Run 6")
r130_end = r130_notes.find("## Run 11", r130_start)
print(r130_notes[r130_start:r130_end][:4200])

print("\n=== LATEST LATENT CONSTRUCTION NOTES ===")
r130_start = r130_notes.find("## Run 88")
r130_end = r130_notes.find("## Run 92", r130_start)
print(r130_notes[r130_start:r130_end][:2200])

print("\n=== COMPRESSION VERSUS ORIGINAL ===")
for r130_n, r130_t in candidates90[0.0].items():
    r130_original = load_original(r130_n)
    if not isinstance(r130_original, torch.Tensor):
        r130_original = torch.as_tensor(r130_original)
    r130_diff = r130_t.detach().cpu() - r130_original.detach().cpu()
    print(r130_n, "norm", round(float(r130_diff.norm()), 6))

print("\n=== TWO CORRECTION CHANNELS ===")
r130_delta = (
    candidates90[1.0]["z_in.weight"] -
    candidates90[0.0]["z_in.weight"]
).detach().cpu().numpy()
print("nonzero column norms:", [
    (i, round(float(v), 6))
    for i, v in enumerate(np.linalg.norm(r130_delta, axis=0)) if v > 1e-7
])
print("D89 norms:", np.linalg.norm(D89, axis=0))
print("correction versus D89:",
      np.linalg.norm(r130_delta[:, 30:32] - D89))
print("Binnov89 bias:", Binnov89[-1])
print("Binnov89 column norms:", np.linalg.norm(Binnov89[:-1], axis=0))

print("\n=== INNOVATION / LATENT SAMPLE OBJECTS ===")
for r130_n in sorted(list(globals())):
    if any(s in r130_n.lower() for s in
           ["innov", "post66", "token_mu", "z88", "x88", "y88", "features88"]):
        r130_o = globals()[r130_n]
        if isinstance(r130_o, (np.ndarray, torch.Tensor)):
            print(r130_n, tuple(r130_o.shape))
        elif isinstance(r130_o, dict):
            print(r130_n, "keys", list(r130_o)[:10])

print("\n=== ROLLOUT ROWS ===")
print(r128_roll.head(3).to_string(index=False))

with open("/work/notes.md", "a") as r130_file:
    r130_file.write(
        "\n## Run 130: distinguish compression from intervention channels\n"
        "- Candidate baseline rotates/compresses the original latent token; scale-dependent changes must be measured relative to that baseline.\n"
        "- Auditing the two correction channels and their innovation targets before independent calibration.\n"
        "- Python runs: 130; forward passes: 6; no submitted edits.\n"
    )
```