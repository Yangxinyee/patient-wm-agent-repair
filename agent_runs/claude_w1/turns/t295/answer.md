gen_samples needs a random generator, and run_stay takes an optional edit argument. I will repeat the comparison with a seeded generator and no perturbation, and also compare run_rows_wc against Xf for the same stay. A near-zero distance tells me which function produces Xf-space inputs, and so which one to use for fresh perturbed samples.

```python
import numpy as np
from scipy.spatial.distance import cdist
st=sf[0]; ix=np.where(sf==st)[0]
for lab,fn in [('gen_nopert',lambda: gen_samples(G3[st],np.random.default_rng(0),pert=False)),('gen_pert',lambda: gen_samples(G3[st],np.random.default_rng(0),pert=True)),('run_rows_wc',lambda: run_rows_wc([st]))]:
    try:
        out=fn(); print(lab,type(out).__name__, len(out) if hasattr(out,'__len__') else '')
        if isinstance(out,list):
            t=out[0]; print(' elem',[np.shape(a) for a in t][:10] if isinstance(t,(tuple,list)) else np.shape(t))
            W=np.array([o[-1] for o in out],float)
        elif isinstance(out,tuple):
            print(' parts',[np.shape(a) for a in out]); W=np.asarray([a for a in out if np.ndim(a)==2 and np.shape(a)[1]==192][0],float)
        else: W=np.asarray(out,float)
        D=cdist(W[:40],Xf[ix]); print(' W',W.shape,'min dist first 6',np.round(D.min(1)[:6],4),'median',np.round(np.median(D.min(1)),4))
    except Exception as e: print(lab,'err',repr(e)[:200])
```