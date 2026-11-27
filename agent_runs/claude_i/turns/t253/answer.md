The last several replies in the previous session were blocked by the content filter, so I don't know what they produced. From now on I'll keep outputs short: statistics and names, not raw listings.

**Where things stand:** no tensors are edited yet. The original model's dose effects are about 6 to 12 times weaker than the effects I estimated from the records:

| Readout | Original model | Kernel-implied target (sel1) |
|---|---|---|
| Insulin x2 | −1.85 | −12.5 |
| Insulin x1/2 | +1.07 | +6.25 |
| NE x2 | +0.36 | +4.3 |
| NE x1/2 | −0.06 | −2.2 |

**Plan:** build a new pathway from the dose deviation (rate minus policy-expected rate) through spare z_post units into a rarely used z dimension. The model's own reading dynamics then carry the effect forward, and I calibrate the scale with dose_edit runs.

**This step:** check which helper functions and arrays are still in the kernel and what my /work files define. The step works if `core_step`, `zpost2`, `eu2` and the fwd_3 state arrays are still present. If they are missing, I must reload them from /work files.

```python
import types, re
g = {k: v for k, v in globals().items() if not k.startswith('_')}
fns = sorted(k for k, v in g.items() if isinstance(v, types.FunctionType))
print(len(fns), ' '.join(fns)[:1200])
arrs = sorted((k, tuple(v.shape)) for k, v in g.items() if hasattr(v, 'shape') and hasattr(v,'dtype') and len(getattr(v,'shape',()))>0)
print(len(arrs)); print('; '.join(f"{k}{s}" for k, s in arrs)[:1500])
for f in ['sim.py','jac2.py','enc2.py','head2.py']:
    s = open('/work/'+f).read()
    print(f, len(s), [m for m in re.findall(r'\ndef (\w+\([^)]*\))', '\n'+s)][:12])
```