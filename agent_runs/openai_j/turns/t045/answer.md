Dose innovations are detectable in the posterior latent state, but barely in the value-head features. I’ll make a reversible probe: double only the numerical dose/rate embedding columns, leaving measurement inputs unchanged. Better rollout responses with little factual-NLL change would support amplifying this pathway; factual degradation or unstable responses would reject it.

```python
import numpy as np
import torch
from pathlib import Path

assert edited() == [], "Expected no existing edits before this probe."
numeric_original = load_original("embed.numeric.weight")
numeric_probe = numeric_original.clone() if torch.is_tensor(numeric_original) else numeric_original.copy()
numeric_probe[:, 2] *= 2.0
numeric_probe[:, 4] *= 2.0
save_edit("embed.numeric.weight", numeric_probe)

probe_name = "numeric_dose_rate_x2"
validation_stays = [int(x) for x in tuning_ids[:100]]

print("=== REVERSIBLE PROBE ===")
print("Edited tensors:", edited())
print("Validation stays:", validation_stays)
print("Changed columns: z_dose (2), z_rate (4); all other columns unchanged.")
w0 = np.asarray(numeric_original)
print("Original column norms:", np.linalg.norm(w0, axis=0))
print("\n=== POTENTIAL SELECTIVE PATHS ===")
for name0 in NAMES:
    if any(s0 in name0 for s0 in [
        "z_post", "z_in", "core.b_in", "core.c_out",
        "core.prelude", "embed.categorical.rate_unit",
        "embed.categorical.dose_unit", "embed.categorical.unit"
    ]):
        print(name0, SHAPES[name0])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 39: first reversible weight probe\n"
            "- Posterior z retains some innovation signal (held-out R2 "
            "~0.10 insulin, ~0.08 NE); value hidden does not. This supports "
            "probing action transmission rather than fitting an innovation "
            "readout to unsupported features.\n"
            "- Temporarily doubled embed.numeric.weight columns 2 (z_dose) "
            "and 4 (z_rate), leaving measurement numerical inputs untouched.\n"
            "- Next: paired original/edited factual validation on first 100 "
            "tuning stays, followed by rollout if useful. This is a probe, "
            "not an accepted repair. Five forward passes used.\n")
```
