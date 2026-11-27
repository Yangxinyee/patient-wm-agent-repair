The last run (202) failed on a bug in my own helper `means_from`. A 64-wide array went where the 17-class bin probabilities were expected. Nothing in the model changed.

**Where things stand**
- **Edit in place:** the saved edit is still the z-channel "writer" (z_post units 55 and 109). My notes show its signal barely reaches the prediction heads, so that design is a dead end. I will replace or drop it.
- **True effects from the records:**
  - insulin x2 → about −13.8 on glucose (model gives −2.15);
  - insulin x1/2 → +6.9 (model +1.23);
  - norepinephrine x2 → +4.6 on MAP (model +0.34);
  - norepinephrine x1/2 → −2.3 (model −0.06).
- **The core problem:** I need a pathway from the dose input to the glucose and MAP predictions that is strong, roughly linear in dose and decays over the following hours. My notes do not record exactly how the state is updated and decays between bundles.

**Next step**
I will re-read the architecture's section structure and take stock of the kernel, with compact output only.

**Hypothesis:** the state has a slot or gated path with a time-flow decay that the heads read strongly. If so, writing a dose signal there gives a controllable, decaying effect. The doc would refute this if every path from the action encoding u into the state goes through heavily mixed nonlinear layers with no such structure.

```python
import os, types
open('/work/notes.md','a').write("\n## Restart 5 (after run 202)\n- run 202 failed (means_from shape bug, 64 vs 17). Writer edit (z_post 55/109) still saved but judged useless. Re-reading ARCHITECTURE.md (state update / slots / decay) to design a stronger dose pathway.\n")
A = open('/model/ARCHITECTURE.md').read()
L = A.split('\n')
print('chars', len(A), 'lines', len(L))
for i,l in enumerate(L):
    if l.startswith('#'): print(i, l[:90])
inv=[]
for k,v in list(globals().items()):
    if k.startswith('_'): continue
    if hasattr(v,'shape') and hasattr(v,'dtype'):
        inv.append(f"{k}{tuple(v.shape)}")
    elif isinstance(v, types.FunctionType):
        inv.append(f"{k}()")
s=' '.join(inv)
print(len(inv)); print(s[:2500])
```