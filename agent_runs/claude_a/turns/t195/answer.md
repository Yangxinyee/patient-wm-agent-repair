The last run measured how strongly each prelude attention head attends to the drug token (u) in each kind of bundle:
- At insulin bundles, head 3 attends u with weight 1.0 in every slot. It is a clean channel for the insulin rate.
- At norepinephrine-only bundles, heads 0 and 3 attend u at about 0.98 or more. So an insulin decoder on head 3 must give about zero on NE encodings.
- At measurement bundles, head 3 hardly attends u (0.001). It reads other tokens there, which is why the write needs a gate.

This confirms the planned write path: head 3, then a staging dim, then a gated write into a memory dim.

**Plan.** The conversation restarted and the function sources were lost, so I will rebuild in small, checked steps.
1. Insulin first. Memory dim gets a gated write of the rate at insulin bundles. The readout is a code-gated value-MLP unit, with a regression on q to compensate the factual mean.
2. Calibrate the gain and decay with the offline rollout replica, against my targets on sel1: x2 −14.3, x1/2 +7.15.
3. Check the factual value NLL offline before spending forward passes.

**This step.** Inventory what survives in the kernel and in /work/lib1.py and lib2.py, with short output only.

**What would change the plan.** If the kill/prelude/tf_loop helpers or the rollout replica are missing, I must rebuild them before any wiring.

```python
import re, os
from collections import Counter
for f in ['/work/lib1.py','/work/lib2.py']:
    s=open(f).read()
    print(f, len(s), 'chars; defs:', ', '.join(m for m in re.findall(r'^def (\w+)\(', s, re.M)))
fn=sorted(k for k,v in globals().items() if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__')
print('kernel funcs', len(fn), ' '.join(fn))
dd=sorted(k for k,v in globals().items() if isinstance(v,dict) and not k.startswith('_') and len(v)>=100)
print('weight-like dicts', dd)
print('prefixes', dict(Counter('.'.join(n.split('.')[:2]) for n in NAMES)))
```