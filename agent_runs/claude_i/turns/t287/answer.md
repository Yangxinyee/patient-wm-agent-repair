The last run confirmed that my numpy reproduction of one model step matches the tool's stored states. The output-vector error was about 5e-3 and the hidden-state error about 0.23. The run then stopped at an undefined name (`ctx_next`), a small bug. Several earlier replies were cut off, so I have lost track of which helper functions exist in the kernel.

Plan, following my notes:
- Repurpose z dimension 18 for insulin and 28 for norepinephrine. The dead hidden units of the z_post network will compute the dose deviation: the recorded rate, decoded from the drug token u, minus the policy-expected rate, decoded from the context. This is gated to bundles that contain the drug.
- Redesign the matching z_in columns so that this deviation shifts the model's glucose or MAP belief.
- Calibrate the scale with dose_edit and check the factual NLL.

Subtracting the policy-expected rate keeps the factual predictions intact, because the model already accounts for the typical dose. It also gives the full structural response under an intervention.

First I need a compact list of what is already built: function definitions in the /work files, the probe dataset and the key kernel variables. If the needed pieces exist (zpost2, eu2, core_step, value_mean, the probe data), I can implement the dead-unit feature in the next run. If they are missing, I will rebuild them from enc2.py.

```python
import os, json, numpy as np
for f in ['enc.py','enc2.py','head.py','head2.py','jac2.py','sim.py']:
    src = open('/work/'+f).read()
    defs = [l.split('(')[0][4:] for l in src.splitlines() if l.startswith('def ')]
    print(f, len(src), defs)
d = np.load('/work/probe_ds.npz'); print('probe_ds', {k: d[k].shape for k in d.files})
s = json.load(open('/work/sel.json')); print('sel', {k:(len(v) if hasattr(v,'__len__') else v) for k,v in s.items()})
fns = sorted(k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__')
print('funcs', fns)
arrs = sorted((k, v.shape) for k,v in globals().items() if isinstance(v, np.ndarray) and v.size>1000)
print('arrays', arrs[:60])
print([n for n in NAMES if 'z_post' in n or 'z_in' in n or 'zpost' in n])
```