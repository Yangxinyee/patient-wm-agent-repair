Run 185 showed that a linear discriminant on the collected features separates drug bundles from other bundles cleanly (drug minimum about 0.79, others at most about -0.58). The drug residual (rate minus policy mean) can only be read back moderately well from those features (R² about 0.69–0.75 for insulin, 0.49–0.59 for norepinephrine). Block 1's feed-forward layer has nearly unused units (513, 121, 650, …) whose contribution is about 1% of the median, so they could be repurposed for the injection.

The notebook has no entries for runs 179–185, so I don't remember exactly what is built in the kernel. Next I will list the kernel's functions and arrays and pull the key g-estimation lines from the start of the notebook. Output stays short. If the hooked core, collected features and calibration targets are all there, I continue the injection design; if pieces are missing, I rebuild only those.

```python
import types, inspect, numpy as np, torch, re
g = dict(globals())
fns, arrs, oth = [], [], []
for k, v in g.items():
    if k.startswith('_') or k in ('In','Out','exit','quit','get_ipython'): continue
    if isinstance(v, types.ModuleType): continue
    if isinstance(v, types.FunctionType):
        try: sig = str(inspect.signature(v))
        except Exception: sig = '()'
        fns.append(f"{k}{sig[:40]}")
    elif isinstance(v, (np.ndarray, torch.Tensor)):
        arrs.append(f"{k}{tuple(v.shape)}")
    elif isinstance(v, (int, float, bool)):
        oth.append(f"{k}={str(v)[:12]}")
    elif isinstance(v, (list, tuple, dict)):
        oth.append(f"{k}:{type(v).__name__}{len(v)}")
print("FNS", len(fns)); print(' | '.join(fns)[:2200])
print("ARR", len(arrs)); print(' '.join(arrs)[:1600])
print("OTH", len(oth)); print(' '.join(oth)[:900])
txt = open('/work/notes.md').read()
print("notes chars", len(txt))
lines = [l for l in txt.split('\n') if re.search(r'g-est|insulin ->|ins ->|policy|lag', l, re.I)]
print('\n'.join(l[:220] for l in lines[:12]))
```