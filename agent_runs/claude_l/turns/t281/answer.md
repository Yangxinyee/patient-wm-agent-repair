Several results from the last run:
- **Expected dose is the bottleneck.** Random gelu features add nothing over a linear decoder. The expected insulin dose given presence stays at R2 ≈ 0.53, and norepinephrine at 0.33–0.38.
- **The state is not lossy about readings.** The last glucose (0.98) and last MAP (0.97) are read cleanly from the context. The previous insulin and norepinephrine rates are only partly retained (0.68 and 0.56). So the state does not fully hold the policy's inputs.
- **Gating is half solved.** The "any drug present" indicator is nearly exact (error 0.004), but per-drug presence indicators are not (errors 0.15–0.2).

The decisive question is whether a residual defined against the context's expected dose is still what the model is missing. If the model's teacher-forced prediction error regresses on r_ctx = dose − expected dose(ctx), with coefficients near the gaps found before (about −1.4 for insulin→glucose and about +3 for norepinephrine→MAP), then adding exactly that term improves NLL. If the coefficients are much smaller, the residual is mostly leaked policy variation and would cost NLL.

The same regression also gives:
- the coefficients at lags 2–3, which show how much memory is still missing;
- the expected-dose terms, which test whether the model has absorbed the policy;
- the error sd per outcome, which sets the NLL cost of overshooting.

```python
import numpy as np, time
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
t0=time.time()
def cvpred(X,y,alpha):
    p=np.zeros_like(y)
    for a,b in KFold(5,shuffle=True,random_state=0).split(X): p[b]=Ridge(alpha=alpha).fit(X[a],y[a]).predict(X[b])
    return p
pi_i=np.zeros(len(DD)); pi_n=np.zeros(len(DD))
pi_i[hi]=cvpred(C[hi],ri[hi],0.1); pi_n[hn]=cvpred(C[hn],rn[hn],0.001)
dmap={}
for j,d in enumerate(DD): dmap.setdefault(d['s'],[]).append(j)
REC=[]
for s,B_ in TRB.items():
    R,_=tf_stay(Wb,B_)
    js=dmap.get(s,[]); td=np.array([DD[j]['t'] for j in js])
    for r in R:
        md,iq=MED[r['code']]; y=r['z']*iq+md; f=np.zeros(12)
        for j,tt in zip(js,td):
            if tt>=r['t']: continue
            lag=int(np.ceil(r['t']-tt-1e-9))
            if 1<=lag<=3:
                if hi[j]: f[lag-1]+=ri[j]-pi_i[j]; f[6+lag-1]+=pi_i[j]
                if hn[j]: f[3+lag-1]+=rn[j]-pi_n[j]; f[9+lag-1]+=pi_n[j]
        REC.append((r['code'],y-r['mean'],*f))
REC=np.array(REC); print('n',len(REC),'t',round(time.time()-t0))
print('err sd by code',[(c,round(REC[REC[:,0]==c,1].std(),2)) for c in sorted(set(REC[:,0].astype(int)))])
names=['ri1','ri2','ri3','rn1','rn2','rn3','pi1','pi2','pi3','pn1','pn2','pn3']
for grp,codes in [('gluc',[CODE_ID['glucose_finger'],CODE_ID['glucose_lab']]),('map',[CODE_ID['map_arterial'],CODE_ID['map_cuff']])]:
    m=np.isin(REC[:,0],codes); y=REC[m,1]; X=np.c_[np.ones(m.sum()),REC[m,2:]]
    b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; s2=e@e/(len(y)-X.shape[1]); se=np.sqrt(np.diag(s2*np.linalg.pinv(X.T@X)))
    print(grp,'n',m.sum(),' '.join(f'{n}:{bb:.2f}({ss:.2f})' for n,bb,ss in zip(names,b[1:],se[1:])))
```