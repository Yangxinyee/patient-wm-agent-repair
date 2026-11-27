The re-encoding is exact (error 0.0 for both e and u), so I can make dose-edited copies of the data for fitting. dsE has 3,388 insulin bundles and 2,292 NE bundles (892 with both), out of 13,036. The `kind` column is 0 for every event, so measurement bundles have to be identified by event name.

**A leak I had missed.** `kill()` leaves the prelude's shared `norm1` reading the killed dims. Once a memory dim holds a value, the prelude attention would read it from the slot tokens and disturb everything downstream. So the scaffold must also zero the prelude `norm1` gain and bias on the circuit dims. That costs the information e and u carry in those dims, but it isn't in my earlier cost measurements.

**Scaffold for this run.** Three killed dims only:
- jR, the reference dim, kept at zero;
- jI, the insulin memory;
- jN, the norepinephrine memory.

On top of `kill()`, the scaffold zeroes:
- prelude `norm1` on those dims;
- the `query_norm` bias;
- the value-MLP input columns, so no existing unit reads them.

I will measure the factual value NLL of two candidate dim sets against the original, on dsE and dsF (300 training stays each).

How I will read it:
- At or below about +0.0015 on average, there is room for the 8 repurposed units and the readout.
- If it is higher, I will try other cheap dims or mean-ablation.

```python
src = r'''
import torch, numpy as np
from lib1 import kill, cl
def scaffold(P0, J):
    P = kill(P0, J, mem=J)
    pre = 'core.prelude.0'
    P[pre+'.norm1.weight'][J] = 0; P[pre+'.norm1.bias'][J] = 0
    P['query_norm.bias'][J] = 0
    P['attr_heads.value_bin.0.weight'][:, J] = 0
    return P
'''
open('/work/lib3.py','w').write(src)
import sys, importlib; sys.path.insert(0,'/work')
import lib3; importlib.reload(lib3)
base={'E':float(tf_eval(PO,dsE,False)[0].mean()),'F':float(tf_eval(PO,dsF,False)[0].mean())}
for J in ([4,67,2],[4,67,108]):
    P1=lib3.scaffold(PO,J)
    c={k:float(tf_eval(P1,d_,False)[0].mean())-base[k] for k,d_ in (('E',dsE),('F',dsF))}
    print(J, {k:round(v,5) for k,v in c.items()}, 'avg', round(np.mean(list(c.values())),5))
```