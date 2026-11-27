The value head has a compact, editable readout, but the previous inspection missed the latent projection because its name is not under `core.*`. I’ll locate the exact projection and the saved dose-residual branch now. If the projection is near zero while the branch detects dose changes, repairing their connection is worth testing; otherwise I’ll use the matched head features instead.

```python
import numpy as np
from pathlib import Path

print("EXACT LATENT TENSORS")
for name in NAMES:
    if any(s in name for s in ("z_in", "z_head", "z_post", "z_prior")):
        w = np.asarray(load_original(name))
        print(name, w.shape, "rms", float(np.sqrt(np.mean(w*w))),
              "max", float(np.max(np.abs(w))))
        if "z_in" in name and w.ndim == 2:
            print("column norms", np.round(np.linalg.norm(w, axis=0), 5))

print("\nPOSTERIOR BRANCH CACHES")
for k, v in list(globals().items()):
    if k.startswith("rp_") and k not in {
        "rp_meta", "rp_owner_list", "rp_owner", "rp_atoms", "rp_drug",
        "rp_end", "rp_encodings", "rp_raw_e", "rp_raw_u",
        "rp_obs_mask", "rp_action_mask", "rp_cf_hidden"
    }:
        if hasattr(v, "shape"):
            print(k, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, list(v)[:12])
        elif isinstance(v, (float, int, str, tuple)):
            print(k, str(v)[:180])
        else:
            print(k, type(v).__name__)

print("\nHEAD CACHE")
for tag, cache in (("original", r164_h0), ("5x", r164_h5)):
    print(tag, {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                for k, v in cache.items()})
print("physical latent directions",
      {k: (np.asarray(v).shape, float(np.linalg.norm(np.asarray(v))))
       for k, v in r168_physdirections.items()})

print("\nRECENT DIRECT-PATH NOTES")
text = Path("/work/notes.md").read_text()
start = text.find("## Run 183")
print(text[start:start+2800])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 192\nLocated exact latent tensor names and recovered posterior branch arrays and matched value-head cache layouts. No edits; testing whether a latent carrier can be connected rather than merely amplified.\n")
```
