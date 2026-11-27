The cached features offer a possible posterior-only probe, although their counterfactual decoding is imperfect. I’ll check their units and coefficients before using them. A useful edit would amplify dose-dependent latent changes while keeping factual perturbations small; poor factual fit or unchanged rollouts would refute this approach.

```python
import inspect
import numpy as np
import torch

for fname in ["fit_post_contrast", "repair_atom_vectors",
              "repair_set_encode", "latent_context_at"]:
    f = globals()[fname]
    print(fname, inspect.signature(f))
    print("  referenced names:", f.__code__.co_names)

print("\nPOSTERIOR DECODERS:")
for drug in ["insulin", "norepinephrine"]:
    key = (drug, "innovation", 10.0, 100.0)
    obj = post_full_models[key]
    print(drug, "model type:", type(obj).__name__)
    if isinstance(obj, dict):
        print([(k, getattr(v, "shape", None), type(v).__name__)
               for k, v in obj.items()])
    elif isinstance(obj, (tuple, list)):
        print([(type(v).__name__, getattr(v, "shape", None)) for v in obj])
    else:
        print("shape:", getattr(obj, "shape", None))
    d = np.asarray(post_latent_directions[drug])
    print("direction shape/norm:", d.shape, np.linalg.norm(d))

print("\nPOSTERIOR FEATURE AND CONTRAST TYPES:")
print("post_H:", type(post_H).__name__, post_H.shape)
for key in [("insulin", 2.0), ("norepinephrine", 2.0)]:
    obj = post_cf[key]
    print(key, type(obj).__name__, getattr(obj, "shape", None))
    if isinstance(obj, dict):
        print([(k, getattr(v, "shape", None)) for k, v in obj.items()])

print("\nRATE SUMMARY:")
print(pd.DataFrame(post_rates, columns=["insulin", "norepinephrine"])
      .describe().round(4))
print("\nNORMALISATION:")
for drug in ["insulin", "norepinephrine"]:
    print(drug, repair_norms[(drug, "rate")])
print("\nRECENT TORCH ARRAYS:")
print([(k, tuple(v.shape)) for k, v in list(globals().items())
       if isinstance(v, torch.Tensor)][-25:])
```
