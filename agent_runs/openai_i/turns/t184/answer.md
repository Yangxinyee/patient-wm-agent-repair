Changing activation conventions did not resolve the mismatch. One remaining architectural distinction matters here: the model may use its top-level `out_norm` for posterior context, rather than `core.out_norm`.

I’ll check that distinction, verify the cached weights against the originals, and separate drug bundles from other bundles. A close match on drug bundles is the key validation for fitting the proposed dose feature.

```python
import numpy as np
import torch
import torch.nn.functional as F

print("=== WEIGHT-INTEGRITY CHECK ===")
for _n in ("z_post.0.weight", "z_post.0.bias", "z_post.2.weight", "z_post.2.bias"):
    _orig = torch.as_tensor(load_original(_n), dtype=torch.float32)
    print(_n, "max difference", float(torch.max(torch.abs(W[_n] - _orig))))

print("\n=== NORMALIZATION TENSORS ===")
print([n for n in NAMES if n.endswith("norm.weight") and ("out" in n or "." not in n)])
print("Readout constants:", readout153.__code__.co_consts[:30])
for _n in ("active152", "valid152", "context", "scores152"):
    if _n in globals():
        _o = globals()[_n]
        print(_n, type(_o).__name__,
              tuple(_o.shape) if hasattr(_o, "shape") else str(_o)[:350])

drug167 = np.any(np.asarray(rates152) > 0, axis=1)[ix166]
contexts167 = {}
for _norm in ("out_norm", "core.out_norm"):
    if _norm + ".weight" not in W:
        continue
    contexts167[_norm] = (
        S166[p166, :192]
        + norm136(hf166.mean(1), _norm)
        + mlp136((torch.log(dt166) / 10)[:, None], "gap")
    )
if "context" in globals() and hasattr(context, "shape") and tuple(context.shape) == (len(S), 192):
    contexts167["retained_context"] = torch.as_tensor(context, dtype=torch.float32)[ix166]

print("\n=== POSTERIOR ERRORS BY CONTEXT AND BUNDLE ===")
scores167 = []
for _cn, _cx in contexts167.items():
    _xx = torch.cat((_cx, E166[ix166], U166[ix166]), dim=1)
    _zz = mlp136(_xx, "z_post")[:, :32]
    _sq = ((_zz - S166[ix166, -32:])**2).mean(1).numpy()
    _r = (
        float(np.sqrt(_sq.mean())), _cn,
        float(np.sqrt(_sq[drug167].mean())),
        float(np.sqrt(_sq[~drug167].mean())),
        int(drug167.sum())
    )
    scores167.append(_r)
    print("overall RMS, context, drug RMS, other RMS, drug count:", _r)

print("\n=== ACTION RATE SUMMARY ===")
print("positive counts:", (rates152 > 0).sum(0))
for _j in range(2):
    _r = rates152[rates152[:, _j] > 0, _j]
    print(_j, "positive-rate quantiles:", np.quantile(_r, [0, .1, .5, .9, 1]))
```
