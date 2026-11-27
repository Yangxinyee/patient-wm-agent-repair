The earlier record analyses found consistent dose-innovation effects, but those estimates remain conditional on the recorded history—not supplied ground truth. I’ll restore the original posterior weights and inspect the reusable helpers. The next candidate must produce a measurable dose response, not merely change an internal latent.

```python
import inspect
import numpy as np
import torch

saved_posterior183 = {n: load(n).clone() if isinstance(load(n), torch.Tensor)
                     else np.array(load(n), copy=True)
                     for n in edited() if n.startswith("z_post.")}
for n in list(edited()):
    if n.startswith("z_post."):
        drop_edit(n)
print("ACTIVE EDITS", edited())

lines183 = []
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            signature183 = str(inspect.signature(v))
        except Exception:
            signature183 = "?"
        lines183.append(k + signature183)
print("\nHELPERS\n" + "\n".join(lines183)[:3900])

lines183 = []
for k, v in list(globals().items()):
    if k.startswith(("rp_", "r168_", "rr_input", "rr_state", "ds_")):
        if hasattr(v, "shape"):
            desc183 = str(tuple(v.shape))
        elif isinstance(v, dict):
            desc183 = "keys=" + str(list(v)[:8])
        elif callable(v):
            desc183 = "callable"
        else:
            desc183 = type(v).__name__
        lines183.append(k + ": " + desc183)
print("\nCACHES\n" + "\n".join(lines183)[:3100])

print("\nCORE SHAPES\n" + "\n".join(
    f"{n}: {SHAPES[n]}" for n in NAMES
    if n.startswith("core.") and "prelude" not in n)[:1900])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 183\nPreserved gain-200 posterior tensors in saved_posterior183, "
            "then restored original z_post weights because the route is ineffective. "
            "All weights are now original. Bounded recovery of helpers and state caches.\n")
```