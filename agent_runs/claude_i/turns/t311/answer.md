Several earlier outputs were blocked for length, so this session will print only short summaries and statistics. I have 35 Python runs left, so each run has to do a lot.

**Where things stand (from the notebook):**
- The true effects are known from the records: about −12.5 for insulin ×2, +6.25 for insulin ×½, +4.33 for norepinephrine ×2 and −2.17 for norepinephrine ×½ on the sel1 stays.
- The original model is 6–12× too weak.
- The state carries almost none of the part of the dose that varies independently of the patient's condition.

**Plan:**
- Use some of the 50 dead hidden units in the z-posterior network to compute the dose deviation (recorded rate minus the rate the policy would give for the current state).
- Send that deviation through a repurposed z dimension and its z_in column into the state update. That should shift the next reading's prediction in the "lower glucose" or "higher MAP" direction.
- Calibrate the strength against the known targets with dose_edit.

**This step:** list the helper functions, arrays and weights already in the kernel and in /work. I need to know how my state-update code (`core_step`) reads its weights, so I can test a changed z_in column locally. The plan fails if the z-posterior or z_in tensors are not where my notes say, or if `core_step` has no weights I can swap in.

```python
import re, os, types, numpy as np
g = dict(globals())
fn = sorted(k for k,v in g.items() if isinstance(v, types.FunctionType) and not k.startswith('_'))
arr = sorted((k, tuple(v.shape)) for k,v in g.items() if isinstance(v, np.ndarray) and not k.startswith('_'))
dct = sorted((k, len(v)) for k,v in g.items() if isinstance(v, dict) and not k.startswith('_') and k not in ('In','Out'))
print('FUNCS:', ' '.join(fn)[:1400])
print('ARRAYS:', '; '.join(f'{k}{s}' for k,s in arr)[:1800])
print('DICTS:', dct[:30])
for k in ['core_step','zpost2','eu2','value_mean','pred_mean_s','view_ctx','zprior','step_err','set_enc2','emb_atoms','ctx2','gap2','pibar']:
    f = g.get(k)
    if isinstance(f, types.FunctionType):
        c = f.__code__
        print(k, c.co_varnames[:c.co_argcount], '| glob:', list(c.co_names)[:20])
for fp in sorted(os.listdir('/work')):
    if fp.endswith('.py'):
        s = open('/work/'+fp).read()
        print(fp, len(s), [m[0] for m in re.findall(r'^def (\w+)\(([^)]*)\)', s, re.M)][:15])
zn = [(n, SHAPES[n]) for n in NAMES if re.search(r'z_|post|prior|zin', n)]
print('ZT:', zn[:30])
```