The brief outputs fill in the picture.
- `run_stay`, `eval_stay_rel` and `step_at` all start each step from the model's true previous state, taken from the fwd_2 states SX. So they are exact one-step simulators, which is enough to measure what an edit changes.
- `pred_mean_gf(y, o, dt, lastval, age)` gives the head's predicted gf mean from a state.
- `fit_psi` uses NEG (31206 = 5201 bundles × 6 non-dose prelude tokens) together with action tokens.

So in Session 9 I was fitting a read ψ on the normalised prelude tokens: it should equal the infusion rate on the dose token and be about 0 on every other token. If a clean ψ exists, head2's mixed attention stops mattering. I would plant ψ as one replaced value dimension of head2, plus one out-projection column. That replacement value dimension is the same single slot needed to keep the factual fit close to the original. I don't remember the result, so I'll redo the fit with clear meanings.

Test: if the ridge fit gives small error on dose tokens (sd < ~0.1) and near-zero spread on NEG tokens, the stage-1 read is viable without touching attention. If NEG spread stays large, I fall back to the attention-key edit.

```python
import numpy as np
print('pa', pa.shape, np.round(pa[:4],3).tolist(), 'colmeans', np.round(pa.mean(0),3).tolist())
for nm,A in [('U_act',U_act),('U0',U0),('NEG',NEG)]:
    nr=np.linalg.norm(A,axis=1); print(nm, A.shape, 'norm mean %.3f sd %.3f'%(nr.mean(),nr.std()), 'row-sd mean %.3f'%A.std(0).mean())
g1=Wt['core.prelude.0.norm1.weight'].numpy(); b1=Wt['core.prelude.0.norm1.bias'].numpy()
print('norm1 gamma mean %.3f, |beta| %.3f'%(np.abs(g1).mean(), np.linalg.norm(b1)))
def ridge_read(Xp,yp,Xn,lam_n=1.0,lam=1e-3):
    X=np.vstack([Xp,Xn]); y=np.r_[yp,np.zeros(len(Xn))]; w=np.r_[np.ones(len(Xp)),lam_n*np.ones(len(Xn))]
    Xa=np.c_[X,np.ones(len(X))]; G=(Xa*w[:,None]).T@Xa+lam*np.eye(Xa.shape[1]); c=np.linalg.solve(G,(Xa*w[:,None]).T@y)
    rp=Xp@c[:-1]+c[-1]-yp; rn=Xn@c[:-1]+c[-1]
    return c, rp.std(), np.abs(rn).mean(), rn.std(), np.abs(rn).max()
for j in range(pa.shape[1]):
    c,sp_,mn,sn,mx=ridge_read(U_act,pa[:,j],NEG)
    print('col',j,'act resid sd %.3f | NEG mean|.| %.3f sd %.3f max %.3f | target sd %.3f'%(sp_,mn,sn,mx,pa[:,j].std()))
```