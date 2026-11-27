fwd_5 tested edit v1 with dose_edit, and v1 did nothing. Its shifts (−0.349, +0.209, +0.018) match the original model's (−0.348, +0.208, +0.018) to three decimals, against targets of about −13, +6.5 and +5.

So the z_post injection is not used where it matters. My leading hypothesis is that in the rollout, bundles after hour 12 draw z from the prior rather than the posterior. In that case the dose reaches the future only through the other path my notes mention ("prelude") or the prior network, and z_post edits can only touch hours ≤12.

To check this I will search ARCHITECTURE.md for how generation and rollout work: prior, posterior, prelude, sampling and drug bundles. I will print only short matching excerpts. If the text says rollout bundles use the prior or prelude path, the hypothesis is confirmed and I must move the injection there. If it says drug bundles in the rollout use z_post, then v1 failed for a different reason, such as gates not firing or the edit not being loaded.

```python
txt = open('/model/ARCHITECTURE.md').read()
print(len(txt))
lines = txt.split('\n')
import re
keys = ['rollout','roll','prior','prelude','generat','sample','draw','posterior','z_prior','dose_edit','hidden','teacher']
for i,l in enumerate(lines):
    ll = l.lower()
    if any(k in ll for k in keys):
        print(i, l[:220])
```