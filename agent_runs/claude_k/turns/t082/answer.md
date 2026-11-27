The architecture description clears up the readout path. The value head reads `with_code = q + code embedding + "last" register`. It applies Linear → GELU → Linear and adds a code × last-bin bias. It then takes a softmax over 17 classes (class 0 means "no value", 1..16 are fine bins). The predicted mean is Σ p(b)·centre[b].

Three consequences for the edit:
- The "last" register only holds earlier values of the same code. A glucose prediction therefore cannot see the last insulin rate directly. The dose reaches the value head only through the state (u_k → z_post / prelude → h, out → view → q).
- Adding a tilt along the bin centres (β·centre_b) to the logits shifts the predicted mean by about β times the predictive variance. That is the natural form for a mean-shift correction, applied to bins 1..16 only.
- The set encoder ends in a norm, so the state may encode dose nonlinearly. I will therefore check how linear the correction is across x2 and x1/2 when I calibrate.

Last time the variable listing was cut at "gb", so I can't yet see all my helper variables, in particular the reconstruction functions and the reading-level arrays. Next I will list the rest compactly and print the code table and bin centres. I need these to compute the cost of repurposing candidate units (35, 53, 125, 24, …) and to fit the dose probes.

```python
import types, numpy as np
names = []
for k, v in sorted(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType) or k in ('In','Out','exit','quit','get_ipython'):
        continue
    if k < 'gb':
        continue
    if callable(v) and not hasattr(v, 'shape'):
        names.append(f"fn:{k}")
    else:
        shp = getattr(v, 'shape', None)
        if shp is not None:
            names.append(f"{k}{tuple(shp)}")
        elif isinstance(v, (list, dict, set, tuple)):
            names.append(f"{k}[{type(v).__name__}{len(v)}]")
        elif isinstance(v, (int, float, np.floating, np.integer)):
            names.append(f"{k}={v}")
        else:
            names.append(f"{k}:{type(v).__name__}")
print(" | ".join(names))
print("\nCODE:", CODE)
print("codes:", codes)
print("VOCAB keys:", list(VOCAB.keys()))
print("config:", VOCAB.get('config'))
vc = VOCAB.get('value_bin_centres')
if isinstance(vc, dict):
    for k2, v2 in vc.items():
        print(k2, np.round(np.array(v2, dtype=float), 2))
else:
    print(type(vc), np.array(vc).shape if vc is not None else None)
print([n for n in NAMES if n.startswith('attr_heads.value_bin') or n.startswith('embed_transition')])
```