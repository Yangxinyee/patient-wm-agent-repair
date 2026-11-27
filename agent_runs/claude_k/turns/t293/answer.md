Forward pass 11 showed that the second "writer" edit does almost nothing. With the same random numbers, the edited model gives insulin x2 −2.07 (original −2.15), insulin ½ +1.19 (+1.23), norepinephrine x2 +0.24 (+0.34) and norepinephrine ½ −0.08 (−0.06). My estimates of the true effects from the records are about −13.8, +6.9, +4.6 and −2.3.

So the z-channel writer is either not firing in the real computation, or its signal does not reach the glucose and MAP predictions. Writer v1 had the same "buried signal" problem, so the z channel as built does not work.

My next step is a short inventory before I design a different pathway:
- reread the state-update and view equations in ARCHITECTURE.md;
- check what forward passes 9 and 10 contained, since they are not in my notes;
- see exactly which rows and columns my current edit changed;
- list the kernel objects I can reuse.

If the edit diff shows the writer units wired as intended, the failure is in propagation. If it does not, the failure is in the gate.

```python
import os, json, numpy as np, torch
T = lambda x: x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
arch = open('/model/ARCHITECTURE.md').read()
print('ARCH chars', len(arch)); print(arch[:15000])
print('=====')
for k in [9, 10, 11]:
    d = f'/work/fwd_{k}'; fl = sorted(os.listdir(d)); print('fwd', k, fl)
    for f in fl:
        if f.endswith('.json'): print(open(os.path.join(d, f)).read()[:800])
print('=====')
for n in edited():
    a = T(load(n)); b = T(load_original(n)); dd = a - b
    if dd.ndim == 2:
        r = np.where(np.abs(dd).sum(1) > 0)[0]; c = np.where(np.abs(dd).sum(0) > 0)[0]
        print(n, dd.shape, 'rows', r[:12], len(r), 'cols', c[:12], len(c), 'maxabs %.3g' % np.abs(dd).max())
    else:
        r = np.where(dd != 0)[0]; print(n, dd.shape, 'idx', r[:12], 'maxabs %.3g' % np.abs(dd).max())
g = {k: type(v).__name__ for k, v in list(globals().items()) if not k.startswith('_')}
print('VARS', {k: v for k, v in g.items() if v not in ('module', 'function', 'type', 'builtin_function_or_method', 'method')})
print('FUNCS', [k for k, v in g.items() if v == 'function'])
```