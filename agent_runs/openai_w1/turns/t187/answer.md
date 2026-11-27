The architecture confirms that doses affect the recurrent state through both the action encoder and posterior. That could support a persistent correction rather than a readout-only patch. I’ll inspect the cached posterior inputs and earlier findings next; accurate dose decoding and a verified physical-effect direction would support this approach.

```python
from pathlib import Path
import numpy as np, torch

print("=== EARLY NOTEBOOK FINDINGS ===")
print(Path("/work/notes.md").read_text()[:4500])
print("=== LATENT WEIGHTS ===")
for _n in NAMES:
    if any(_s in _n for _s in ("z_post", "z_prior", "z_in", "z_head", "core.gap", "core.flow")):
        print(_n, SHAPES[_n])
print("=== POSTERIOR / DIRECTION CACHES ===")
for _k, _v in sorted(list(globals().items())):
    if _k.startswith(("ds_", "post_", "latent_", "obsdir_", "rr_input", "rr_ctx", "rr_e", "rr_u")):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            print(_k, type(_v).__name__, tuple(_v.shape))
        elif isinstance(_v, dict):
            print(_k, "dict keys:", list(_v.keys())[:6])
        elif callable(_v):
            print(_k, "function args:", _v.__code__.co_varnames[:_v.__code__.co_argcount])
        elif isinstance(_v, (tuple, list)):
            print(_k, type(_v).__name__, "length", len(_v))
        elif isinstance(_v, (str, float, int, bool)):
            print(_k, repr(_v)[:120])
print("=== RECENT OBJECT NAMES ===")
print([_k for _k in globals() if any(_s in _k.lower() for _s in ("direction", "innovation", "decode", "lag_kernel", "recovery"))][:50])
with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 170\nArchitecture supports an action/posterior-to-state edit. Recovering only posterior inputs, physical observation directions, and early causal conclusions; no further weight changes.\n")
```