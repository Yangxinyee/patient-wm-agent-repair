Three results.
1. `u_of` reproduces the stored dose tokens exactly (error 5e-7), so I can build tokens for any doses, including the doubled and halved doses the evaluation uses.
2. The constrained linear rate reads are poor off the factual cases. The insulin read saturates above 4 (rates 6 and 8 read 4.9 and 5.1) and has bin-shaped steps. The norepinephrine read is unreliable. A plain linear read of the normalised dose token would therefore under-count the ×2 effect for high rates.
3. Head2's value output is almost rank-2: singular values 144 and 18, then about 1 and below. The bottom 3 directions carry 1e-9 of its energy, so many value dimensions can be repurposed at no cost.

**Conclusion:** the gate is solved, and the bottleneck has room. The weak link is a rate read that stays linear under ×2 and ×½.

**Before designing a multi-feature nonlinear read,** I'll check whether the model has a more direct path: tensors named `pending_mlp`, `filter_gate`, `kq` and `query_norm` exist, and my notes never describe them. If running infusions feed the prediction heads directly, the channel could be far simpler.

**Test:** if ARCHITECTURE.md shows running infusions (rate and dose) entering the view or with_code, I switch to editing that path. Otherwise I continue with the prelude design.

```python
txt=open('/model/ARCHITECTURE.md').read()
import re
keys=['pending','filter','kq','query_norm','encode_action','size','with_code','view','rate_bin','dose']
lines=txt.split('\n')
shown=set()
for i,l in enumerate(lines):
    if any(k in l for k in keys):
        for j in range(max(0,i-1),min(len(lines),i+2)):
            if j not in shown: shown.add(j); print(j, lines[j][:220])
print(len(lines),'lines')
```