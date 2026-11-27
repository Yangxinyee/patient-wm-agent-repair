The prelude replica is explicit: tokens → norm1 → attention → X1 → norm3 → FFN → X2 → `prelude_norm` → E. So I can add gated FFN units that read norm3(X1) of each slot.

The layout I'll build:
- **Insulin dimensions:** staging jS=4, memory jI=67, shared reference jR=2. These are three of the cheapest dimensions I measured.
- **Base model:** cut those dimensions out of the original. Also zero the prelude norm1/norm3 weights on the memory dimension, so attention and FFN never read the memory.

On that base I'll recompute the teacher-forced internals and fit two linear readouts from the slots' norm3(X1):
- **Gate:** insulin present (+1) vs not (−1).
- **Rate decoder:** insulin rate at drug bundles, with target 0 at norepinephrine-only bundles.

What would confirm feasibility: a gate margin that cleanly separates the two groups (insulin minimum well above the others' maximum), and a rate decoder with R² ≳ 0.95 at insulin bundles that stays near 0 at norepinephrine-only bundles.

```python
import numpy as np, torch
jS,jI,jR=4,67,2
PK=kill(PO,[jS,jI,jR],mem=[jI])
PK['core.prelude.0.norm1.weight'][jI]=0; PK['core.prelude.0.norm3.weight'][jI]=0
_,_,stK=tf_loop(PK,dsE,keep=('hf','E'))
with torch.no_grad(): pr=prelude(PK, stK['hf'], dsE['e'], dsE['u'], stK['z'])
print('E check', float((pr['E']-stK['E']).abs().max()))
n3=pr['n3'][:,:4].numpy()            # B,4,192
B=n3.shape[0]; ins=(ri>0); ne=(rn>0)
X=np.concatenate([n3.reshape(-1,192), np.ones((B*4,1))],1)
rep4=lambda a: np.repeat(a,4)
def ridge_fit(X,y,w=None,lam=1e-2):
    w=np.ones(len(y)) if w is None else w
    A=(X*w[:,None]).T@X+lam*np.eye(X.shape[1]); return np.linalg.solve(A,(X*w[:,None]).T@y)
tg=np.where(rep4(ins),1.,-1.); bg=ridge_fit(X,tg); g=(X@bg).reshape(B,4)
print('gate ins min %.3f  non-ins max %.3f  NEonly max %.3f'%(g[ins].min(), g[~ins].max(), g[ne&~ins].max() if (ne&~ins).any() else np.nan))
m=rep4(ins|ne); bs=ridge_fit(X[m], rep4(ri)[m]); s=(X@bs).reshape(B,4)
res=s[ins]-ri[ins][:,None]
print('rate R2 %.4f resid sd %.3f | NEonly |s| mean %.3f p99 %.3f | rate sd %.3f'%(1-res.var()/ri[ins].var(), res.std(), np.abs(s[ne&~ins]).mean(), np.percentile(np.abs(s[ne&~ins]),99), ri[ins].std()))
```