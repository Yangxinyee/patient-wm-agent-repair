The records support insulin lowering glucose and norepinephrine raising MAP, with different persistence. The saved fits include much less regularized extractors, but their large coefficients alone do not establish usefulness. I’ll measure held-out dose sensitivity against factual variation; that tradeoff determines whether a small, targeted weight edit is feasible.

```python
import numpy as np
import torch
import dis

print("DOSE EXTRACTOR VALIDATION")
for key, b in contrast_fits88.items():
    b = np.asarray(b)
    if b.shape != (768, 2):
        continue
    pred = DX88 @ b
    factual_score = fact_norm88 @ b
    held = ~DT88
    slopes = [
        float(np.sum(pred[held, j] * DY88[held, j]) /
              max(np.sum(DY88[held, j]**2), 1e-12))
        for j in range(2)
    ]
    err = np.sqrt(np.mean((pred[held] - DY88[held])**2, axis=0))
    fsd = np.std(factual_score, axis=0)
    print(key, "held slopes", np.round(slopes, 3),
          "held RMSE", np.round(err, 3),
          "factual SD", np.round(fsd, 3))

print("\nCHOSEN RAW FITS")
for name, b in chosen90.items():
    print(name, "mean object", summary100(means90[name]))
    print("raw factual score mean/SD",
          np.round(np.mean(fact88 @ b, axis=0), 4),
          np.round(np.std(fact88 @ b, axis=0), 4))
    matches = [(str(k), float(np.max(np.abs(b - np.asarray(c)/scale88[:, None]))))
               for k, c in contrast_fits88.items()
               if np.asarray(c).shape == b.shape]
    print("normalization matches", sorted(matches, key=lambda x: x[1])[:2])

print("\nDOSE BLOCK METADATA")
for m in contrast_meta88[:3]:
    print(summary100(m))
block101 = len(DX88)//10
for key in [(1.0, 1e-8), (10.0, 1e-8), 1.0]:
    if key not in contrast_fits88:
        continue
    p = DX88 @ contrast_fits88[key]
    print("factor response:", key)
    for j in range(10):
        sl = slice(j*block101, (j+1)*block101)
        print(j, "target", np.round(DY88[sl].mean(axis=0), 3),
              "predicted", np.round(p[sl].mean(axis=0), 3))

print("\nSTATE/CACHE ARRAYS")
count101 = 0
for k, v in sorted(list(globals().items())):
    if isinstance(v, (np.ndarray, torch.Tensor)) and v.ndim and v.shape[0] in [4213, 4313]:
        print(k, tuple(v.shape))
        count101 += 1
        if count101 >= 35:
            break

print("\nHELPER RETURN OPERATIONS")
for name in ["prelude_cache88", "state_context", "context_with_flow", "state_update82"]:
    f = globals()[name]
    print(name, "dependencies", f.__code__.co_names,
          "constants", f.__code__.co_consts)
    ins = list(dis.get_instructions(f))
    print("return tail:", [(i.opname, i.argrepr) for i in ins[-18:]])

print("\nREADOUT LIVE OBJECTS")
for name in ["row97", "lasts97", "gaps97", "ages97", "bins97", "codes97",
             "centres97", "observed_mean97", "T97"]:
    v = globals()[name]
    if name == "row97":
        print(name, v)
    elif isinstance(v, dict):
        print(name, {k: np.shape(x) for k, x in v.items()})
    elif isinstance(v, list):
        print(name, v[:2])
    else:
        print(name, v[:3])
print("\nCURRENT EDITS", edited())
```
