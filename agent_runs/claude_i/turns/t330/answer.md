The run before the restart failed on a float32/float64 mismatch while I was simulating a state update with a modified insulin token. The four replies after that were blocked, so I'll keep code and output short. 27 Python runs and 47 forward passes remain.

My working hypothesis is that the original model barely responds to exogenous dose variation. Its rollout gives insulin ×2 −2.0 against a kernel-implied −12.5, and norepinephrine ×2 +0.36 against +4.3.

The edit I plan:
- Repurpose dead hidden units of the z-posterior network to compute a gated "dose minus policy-expected dose" signal.
- Feed that signal into one rarely used z dimension per drug.
- Point that dimension's z_in column at a direction that mimics a lower glucose (or higher MAP) reading.
- Calibrate the scale with dose_edit and check factual NLL.

First I need the exact tensor names and shapes of the z-posterior and z_in layers, and which helper functions exist in my kernel and /work files. If z_in is a plain linear layer, the plan is cheap to implement. If it is an MLP, I'll need a different injection point.

```python
import numpy as np, os
with open('/work/notes.md','a') as f:
    f.write("\n## Session 7 (restart after reply 325, 273 runs used, 3 fwd)\n- Plan: z_post dead units -> gated dev=(rate - expected rate) into a spare z dim; z_in column -> 'lower glucose reading' direction; calibrate scale via dose_edit; check factual.\n")
zn=[n for n in NAMES if ('z_post' in n or 'z_in' in n or 'zpost' in n)]
for n in zn: print(n, SHAPES[n])
fns=sorted(k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__')
print(len(fns), ' '.join(fns)[:1200])
for fn in ['sim.py','jac2.py','enc2.py','head2.py']:
    s=open('/work/'+fn).read()
    print(fn, len(s), [l[4:l.find('(')] for l in s.splitlines() if l.startswith('def ')][:20])
```