The architecture confirms that drug effects must pass through the shared recurrent state; the value heads cannot directly read another event’s last dose. I’ll recover the causal estimates and state objects next. Consistent effects across the identification methods would support a dose-sensitive state edit; conflicting estimates would require resolving the causal evidence first.

```python
import torch

print("=== EARLY IDENTIFICATION NOTES ===")
print(Path("/work/notes.md").read_text()[:6400])

print("\n=== STATE / FEATURE OBJECT INVENTORY ===")
inventory67 = []
for k, v in list(globals().items()):
    if k.startswith("_"):
        continue
    if isinstance(v, (np.ndarray, torch.Tensor)):
        shape = tuple(v.shape)
        if len(shape) >= 2 and (
            shape[0] >= 1000 or
            any(s in k.lower() for s in ("state", "hidden", "latent", "post", "ctx"))
        ):
            inventory67.append((k, shape))
    elif isinstance(v, dict) and any(
        s in k.lower() for s in ("state", "bundle", "latent", "post", "tensor", "cache")
    ):
        inventory67.append((k, "dict", list(v)[:8]))
    elif isinstance(v, (list, tuple)) and any(
        s in k.lower() for s in ("state", "bundle", "latent", "post")
    ):
        inventory67.append((k, type(v).__name__, len(v)))
print(str(inventory67)[:2800])
print("\nCurrent edits:", edited())
```
