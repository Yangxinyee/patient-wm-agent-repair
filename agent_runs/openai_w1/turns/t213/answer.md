The architecture confirms that doses can influence predictions through both the action token and the latent token. Earlier tests changed the latent state without changing rollouts, so increasing the same posterior edit again is not justified. I’ll inspect its transmission weights and recover the causal evidence. A weak latent projection would support repairing that projection; an already substantial projection would point us toward another pathway.

```python
import numpy as np
from pathlib import Path

print("EARLY EVIDENCE")
print(Path("/work/notes.md").read_text()[:5400])

print("\nLATENT/PRELUDE WEIGHTS")
for name in NAMES:
    if name.startswith(("core.z_in", "core.z_head", "core.prelude.0.attn", "core.prelude.0.self_attn", "attr_heads.value_bin")):
        w = load_original(name)
        print(name, w.shape, "rms", round(float(np.sqrt(np.mean(w*w))), 6),
              "max", round(float(np.max(np.abs(w))), 6))

print("\nRECENT CACHE NAMES")
for prefix in ("r188", "r187", "rp_", "r168", "r164"):
    found = []
    for k, v in list(globals().items()):
        if k.startswith(prefix):
            if hasattr(v, "shape"):
                found.append((k, tuple(v.shape)))
            elif isinstance(v, dict):
                found.append((k, "keys", list(v)[:12]))
            else:
                found.append((k, type(v).__name__))
    print(prefix, found[:22])

print("\nCARRIER/HEAD VARIABLES")
print([k for k in globals() if any(s in k.lower() for s in
      ("carrier", "physical", "kernel", "head5", "head_5", "contrast", "causal"))][:45])
print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 190\nArchitecture confirms separate action and latent tokens. Inspecting latent transmission and bounded causal/cache recovery; increasing the ineffective posterior alone is rejected. No weights changed.\n")
```
