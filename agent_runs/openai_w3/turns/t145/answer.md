The replay helpers cache original weights, so local calibration will need explicit cache updates when testing an edit. I’ll first compare the two retained dose extractors across all fitted dose factors. Consistent contrasts—and small scores on factual inputs—would support a pilot; large factual scores or saturation would rule one out.

```python
import numpy as np
import torch

def shape132(x):
    if isinstance(x, dict):
        return {str(k): shape132(v) for k, v in list(x.items())[:3]}
    if isinstance(x, (tuple, list)):
        return [shape132(v) for v in x[:8]]
    return (type(x).__name__, tuple(x.shape)) if hasattr(x, "shape") else type(x).__name__

print("=== FIT INPUTS ===")
for nm132 in ("DX88", "DY88", "DT88", "fact88", "hidden88", "Cfact88",
              "Cdelta88", "pool_delta104", "PF89", "payload104",
              "contrast_fits88", "pool_fits89"):
    print(nm132, shape132(globals().get(nm132)))

y132 = as_numpy(DY88)
for label132, x132, beta132 in (
    ("FFN", globals().get("DX88"), globals().get("Bdelta88")),
    ("attention", globals().get("pool_delta104"), globals().get("payload104"))
):
    x132, beta132 = as_numpy(x132), as_numpy(beta132)
    if x132.shape[1] != beta132.shape[0]:
        print(label132, "incompatible shapes", x132.shape, beta132.shape)
        continue
    pred132 = x132 @ beta132
    held132 = ~np.asarray(DT88, dtype=bool)
    print(label132, "RMSE all/held",
          np.sqrt(np.mean((pred132-y132)**2, axis=0)),
          np.sqrt(np.mean((pred132[held132]-y132[held132])**2, axis=0)))
    for col132 in range(2):
        nz132 = held132 & (np.abs(y132[:, col132]) > 1e-8)
        print("drug", col132, "held slope",
              np.sum(pred132[nz132, col132]*y132[nz132, col132]) /
              np.sum(y132[nz132, col132]**2),
              "cross RMS", np.sqrt(np.mean(pred132[nz132, 1-col132]**2)))

print("\n=== FUNCTION OUTPUT SHAPES ===")
for nm132 in ("prelude_cache88", "parts105"):
    out132 = globals()[nm132](h81[:2], e81[:2], u81[:2], z88[:2])
    print(nm132, shape132(out132))

print("\n=== CALIBRATION VARIABLES ===")
for nm132, ob132 in list(globals().items()):
    if nm132.endswith("105") and not callable(ob132):
        print(nm132, shape132(ob132))
```
