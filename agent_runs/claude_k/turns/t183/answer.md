A single linear readout of u can't represent the edited doses. Even with an extreme norm (|p| = 920) the insulin error is 0.07, with a maximum of 1.1 units/h; norepinephrine gives 0.03 (max 0.52). The encoder bends rate into a curved path in u, so a one-unit linear writer won't be accurate for x0–x3.

Two ways forward:
- **Several nonlinear z_post units on u, combined linearly.** The weights would come from a least-squares regression on fixed features of my exact u rebuild, which counts as fitting my own small model.
- **Routing a clean channel through the action encoder.** This is blocked by its final LayerNorm, which rescales any channel by a configuration-dependent σ.

The first is cleaner, but it needs spare z_post units. Units whose weights into the 32 posterior-mean outputs are tiny can be repurposed almost for free, because the log-sd outputs are not used by the state update or the heads. So I'll count units by mean-output vs log-sd weight norms. I'll also try a random-feature GELU decoder to see how accuracy depends on the number of units.

If ≥ 20 units are nearly free and ~20 features reach ≲0.02 error, this route works.

```python
W2o_=np.asarray(W2o); W0o_=np.asarray(W0o); b0o_=np.asarray(b0o)
cm=np.linalg.norm(W2o_[:32],axis=0); cs=np.linalg.norm(W2o_[32:],axis=0)
print('mean-col norm quantiles',np.round(np.quantile(cm,[0,.05,.1,.25,.5,.9]),4),'| units with mean-col<1e-3:',int((cm<1e-3).sum()),'<1e-2:',int((cm<1e-2).sum()))
print('cheapest units by mean-col norm',np.argsort(cm)[:12], np.round(np.sort(cm)[:12],4))
# random-feature decoder on exact U_
from numpy.linalg import lstsq
Uc=U_-U_.mean(0); _,S_,Vt_=np.linalg.svd(Uc*np.sqrt(Wt)[:,None],full_matrices=False)
print('u PCA sv (top 12)',np.round(S_[:12]/S_[0],4))
rng=np.random.default_rng(0)
for K in [8,16,32,64]:
    for kdim in [6,12]:
        Pc=Vt_[:kdim]; Zp=Uc@Pc.T; sd_=Zp.std(0)
        Wk=rng.normal(size=(K,kdim))/sd_/np.sqrt(kdim)*2; ck=rng.uniform(-2,2,size=K)
        F_=gelu_np(Zp@Wk.T+ck); F1=np.c_[F_,np.ones(len(F_)),Zp]
        for nm_,y_ in [('I',YI),('N',YN)]:
            co,_,_,_=lstsq(F1*np.sqrt(Wt)[:,None],y_*np.sqrt(Wt),rcond=None); e_=F1@co-y_
            print(K,kdim,nm_,'rmse',round(np.sqrt(np.average(e_**2,weights=Wt)),4),'max',round(np.abs(e_).max(),3),end=' | ')
    print()
```