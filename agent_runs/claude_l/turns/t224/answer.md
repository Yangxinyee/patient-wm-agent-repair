The attention pattern is clear and very hard:

| Bundle | Head | Winner | Score gap to runner-up (median / 5th pct) |
|---|---|---|---|
| Drug | 0, 1 | u (≈93%), otherwise e | ≈20 / ≈5 |
| Drug | 2 | u always | 24 / 14.5 |
| Drug | 3 | slots | — |
| Non-drug | 3 | u (u_empty) always | 12 / 10 |
| Non-drug | 2 | u or e, about half each | — |
| Non-drug | 0, 1 | mostly e or slot 2; the original z wins in 87 cases | — |

So the swap design is natural. Push u's key far down with a large constant b′, and let a z token that copies u take over every role u had. This includes u_empty at non-drug bundles in heads 2 and 3. The few cases where the original z won will change, as in the earlier copy test, which cost nothing.

Next I'll find the LayerNorm-output direction for the retired token that maximises the worst-case margin: the winning score minus u's score, over all 15,232 query vectors from the 20 stays. I'll start with the mean-direction heuristic and then a linear-programming optimum.

**Success** means a worst-case margin of at least about 15 (attention weight below e⁻¹⁵), with the u-dependent wobble from a finite |b′| checked as well.

```python
gam=W0['core.prelude.0.norm1.weight'].numpy().astype(np.float64); bet=W0['core.prelude.0.norm1.bias'].numpy().astype(np.float64)
Af=AA.reshape(-1,192).astype(np.float64); top=LG.max(-1).reshape(-1).astype(np.float64)
Vs=Af*gam[None]; Vc=Vs-Vs.mean(1,keepdims=True); base_=Af@bet
nh=-Vc.mean(0); nh=nh-nh.mean(); nh=nh/nh.std()
mg=top-base_-Vc@nh
print('heuristic margin min %.1f p1 %.1f med %.1f'%(mg.min(),np.quantile(mg,.01),np.median(mg)))
from scipy.optimize import linprog
idx_=np.random.RandomState(1).choice(len(Af),4000,replace=False)
c_=np.zeros(193); c_[-1]=-1
Aub=np.c_[Vc[idx_],np.ones(len(idx_))]; bub=top[idx_]-base_[idx_]
Aeq=np.r_[np.ones(192),0][None]; 
t0=time.time()
lp=linprog(c_,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=[0],bounds=[(-2.5,2.5)]*192+[(None,None)],method='highs')
n2=lp.x[:192]; n2=(n2-n2.mean())/n2.std()
mg2=top-base_-Vc@n2
print('LP status',lp.status,'%.0fs'%(time.time()-t0),'margin min %.1f p1 %.1f med %.1f'%(mg2.min(),np.quantile(mg2,.01),np.median(mg2)))
nstar=n2 if mg2.min()>mg.min() else nh
```