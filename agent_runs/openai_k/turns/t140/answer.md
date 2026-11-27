The original rollout barely responds to norepinephrine, while insulin has a modest effect. That differs from the record-derived impulse estimates, but the expected rollout shift also depends on the actual rates and persistence. I’ll recover those aggregate targets and the candidate’s latent-channel construction; this will distinguish a calibration problem from incorrect drug routing.

```python
import re, dis
import numpy as np
import torch
from pathlib import Path

print("=== EARLY IDENTIFICATION NOTES ===")
r129_notes = Path("/work/notes.md").read_text()
print(r129_notes[:3300])

print("\n=== LATENT CANDIDATE RECIPE ===")
r129_f = globals().get("build_candidate89")
if r129_f is not None:
    r129_instructions = list(dis.get_instructions(r129_f))
    for r129_i in r129_instructions:
        if r129_i.opname not in ["RESUME", "CACHE", "EXTENDED_ARG"]:
            print(r129_i.opname, r129_i.argrepr)

print("\n=== RETAINED LATENT FIT ARRAYS ===")
for r129_n in sorted(list(globals())):
    if re.search(r"(89|90|126)$", r129_n):
        r129_o = globals()[r129_n]
        if isinstance(r129_o, (np.ndarray, torch.Tensor)):
            print(r129_n, tuple(r129_o.shape))

print("\n=== ACTUAL PROJECTION EDIT ===")
r129_delta = (
    candidates90[1.0]["z_in.weight"].detach().cpu().numpy()
    - load_original("z_in.weight").detach().cpu().numpy()
)
r129_u, r129_s, r129_vt = np.linalg.svd(r129_delta, full_matrices=False)
print("singular values:", np.round(r129_s[:8], 5))
print("column norms:", np.round(np.linalg.norm(r129_delta, axis=0), 4))
print("direction shape:", np.shape(globals().get("direction")))
if np.shape(globals().get("direction")) == (32, 2):
    print("direction column norms:", np.linalg.norm(direction, axis=0))
    print("projection on direction:", np.linalg.norm(r129_delta @ direction, axis=0))

print("\n=== RATE DISTRIBUTIONS ===")
for r129_drug in ["insulin", "norepinephrine"]:
    r129_rate = df.loc[df.event.eq(r129_drug), "rate"]
    print(r129_drug, r129_rate.describe(
        percentiles=[.05, .25, .5, .75, .95]).round(4).to_dict())

with open("/work/notes.md", "a") as r129_file:
    r129_file.write(
        "\n## Run 129: latent edit recipe and exposure scale\n"
        "- Official original tuning rollout: insulin x2 -1.518, half +0.791; NE x2 +0.169, half -0.038.\n"
        "- Recovering latent candidate construction and rate scale to connect causal impulse estimates to rollout targets.\n"
        "- Python runs: 129; forward passes: 6; no submitted edits.\n"
    )
```