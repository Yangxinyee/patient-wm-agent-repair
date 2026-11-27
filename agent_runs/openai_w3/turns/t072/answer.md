The records suggest a substantial insulin effect and a positive norepinephrine effect, unlike the original rollouts. The architecture offers several possible routes from dose to prediction, but the pilot did not activate an effective one. I’ll restore the original weights and inspect the latent and state pathways. Weak latent coupling would support a different injection route; strong coupling would implicate the pilot’s direction or gate.

```python
import numpy as np
import torch
from pathlib import Path

for name in list(edited()):
    drop_edit(name)

def compact_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

print("RESTORED ORIGINAL:", edited())
print("\n=== RELEVANT TENSOR NORMS ===")
for name in NAMES:
    if (name.startswith(("z_in.", "z_head.", "z_prior.", "z_post.",
                         "core.delta_in", "core.a_in", "core.b_in.", "core.c_out.",
                         "attr_heads.value_bin.", "last_numeric."))):
        x = compact_array(load_original(name))
        print(name, x.shape, "rms", round(float(np.sqrt(np.mean(x*x))), 6),
              "max", round(float(np.max(np.abs(x))), 6))

print("\n=== LIVE HELPERS ===")
print(", ".join(sorted(k for k, v in list(globals().items())
                       if callable(v) and not k.startswith("_")
                       and getattr(v, "__module__", "") == "__main__")))

print("\n=== LIVE DATA / FITS ===")
lines = []
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"x", "v", "lines"}:
        continue
    if hasattr(v, "shape") and not callable(v):
        lines.append(f"{k}: {type(v).__name__}{tuple(v.shape)}")
    elif isinstance(v, dict) and any(s in k.lower() for s in
                ["fit", "caus", "effect", "dose", "repair", "weight", "state", "pilot"]):
        lines.append(f"{k}: keys={list(v)[:8]}")
print("\n".join(lines)[:5700])

with Path("/work/notes.md").open("a") as f:
    f.write("\nPython run 63; forwards 6. Pilot reverted: posterior changes had negligible "
            "effect on prediction-driving out. Record-derived reference shifts remain "
            "approximately insulin x2 -12.96, half +6.48; NE x2 +4.28, half -2.14 "
            "on the initial tuning subset. Inspecting latent/state coupling before a new edit.\n")
```
