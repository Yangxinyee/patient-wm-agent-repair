The notebook’s conditional estimates put insulin’s next-hour glucose effect near −1.84 per rate unit and norepinephrine’s MAP effect near +3.2 to +3.6. These remain assumption-dependent targets. The output was truncated before the calibration objects appeared, so I will inspect those directly; this should identify how to recenter the trial without changing its drug coefficients.

```python
import numpy as np
import torch

_g272 = {k: v for k, v in list(globals().items()) if not k.startswith("__")}
print("CALLABLES:", " ".join(k for k, v in _g272.items() if callable(v))[:4500])
print("DICTIONARIES:", [(k, len(v), list(v)[:6]) for k, v in _g272.items()
                       if isinstance(v, dict) and not k.startswith("_g")][-55:])
print("ARRAYS:", [(k, tuple(v.shape)) for k, v in _g272.items()
                 if isinstance(v, (np.ndarray, torch.Tensor))][-70:])

print("\nHEAD AND TRANSPORT SHAPES")
for k in ["attr_heads.value_bin.0.weight", "attr_heads.value_bin.0.bias",
          "attr_heads.value_bin.2.weight", "attr_heads.value_bin.2.bias",
          "core.flow.a", "core.a_in", "core.delta_in"]:
    a = np.asarray(load(k))
    print(k, a.shape)
    if a.ndim == 2 and k.startswith("attr_heads"):
        d = a - np.asarray(load_original(k))
        rows = np.flatnonzero(np.max(np.abs(d), axis=1) > 1e-7)
        print("changed rows:", rows.tolist())
        print("largest changes:", [(int(i), float(np.linalg.norm(d[i]))) for i in rows[:20]])

print("\nLATER IDENTIFICATION / DESIGN NOTES")
for section in _notes271.split("\n## "):
    header = section.splitlines()[0]
    if any(s in header.lower() for s in
           ["target", "causal", "centering", "dose-response", "exposure", "run 184", "run 185"]):
        print("##", section[:1150])
print("\nFRESH STATES:", {k: (v.shape, str(v.dtype)) for k, v in _s271.items()})
```
