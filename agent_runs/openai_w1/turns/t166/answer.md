The original rollouts are nearly insensitive to norepinephrine and show only small insulin responses, unlike the adjusted associations in the records. I’ll check whether the posterior inputs expose both dose and pre-dose context. A usable repair pathway needs to separate unexpected dose variation from the policy-driven part.

```python
import numpy as np, torch
print("LATENT AND PRELUDE TENSORS:")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_prior", "z_head", "z_in", "prelude.0")):
        print(name, SHAPES[name])
print("\nNUMERIC COLUMN NORMS:")
nw = np.asarray(load_original("embed.numeric.weight"))
print(np.linalg.norm(nw, axis=0))
print("PROBE CHANGE NORMS:", np.linalg.norm(np.asarray(he_numeric_probe) - np.asarray(he_numeric_original), axis=0))
print("\nLATER ENCODING CACHE NAMES:")
for name, obj in sorted(list(globals().items())):
    if name.startswith(("rr1", "r10", "r11", "r12", "r13")):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "keys:", str(list(obj)[:8])[:170])
print("\nRECORD METADATA:", {k: ce_meta[k] for k in [("insulin", "dose"), ("insulin", "rate"), ("norepinephrine", "dose"), ("norepinephrine", "rate")]})
print("\nVALIDATION NOTES:")
nt = open("/work/notes.md").read()
for marker in ("## Run 110", "## Run 111", "## Run 112", "## Run 113", "## Run 114", "## Run 115", "## Run 116", "## Run 117", "## Run 118", "## Run 119"):
    start = nt.find(marker + "\n")
    if start >= 0:
        end = nt.find("\n## ", start + 4)
        piece = nt[start:end if end >= 0 else start + 500]
        print(piece[:500])
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 152\nOriginal rollout artifacts confirm under-response, especially NE x2. Investigating a posterior dose-minus-pre-dose-context pathway, instead of unconstrained output amplification. No edits. Counts: 152 Python runs, 8 forward passes.\n")
```