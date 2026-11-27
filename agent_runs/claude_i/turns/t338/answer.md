Both drugs show the same pattern in the factual residuals: the original model misses most of the real dose effect.
- **Finger glucose:** residuals fall with the insulin deviation (rate − pibar): −1.6, −1.8, −1.1 per u/h at lags 1–3 h (se ≈ 0.65).
- **Arterial MAP:** residuals rise with the NE rate: +2.9 per mcg/kg/min at lag 1 (se 0.8). The model systematically under-predicts MAP while NE runs.

So an injection in the causal direction (insulin: −c·deviation; NE: +c·rate, no compensation needed) should improve factual fit, not hurt it. That resolves my earlier worry about NE.

Next I'll build the inputs for the repurposed units:
- **Dose tokens:** u tokens at every drug bundle with the dose scaled ×0.5, 1, 2, 3, using `u_for`, so the decoders stay linear over the edit range.
- **Linear gates:** one per drug, separating bundles that contain each drug.
- **Decoders:**
  - insulin: f·rate − pibar(g) from [ctx, u];
  - NE: f·rate from u.

Success means positive gate margins and decoders with high cross-validated R² (by stay) and small residual sd at f = 1.

```python
import time
with open('/work/notes.md','a') as f_: f_.write("- KEY (run281): factual residuals vs dose: gf ~ dev_ins lags1-3 coef -1.59,-1.77,-1.12 (se .65); ma ~ NE rate lags1-3 +2.86,-0.05,-1.06 (se .8). Model MISSES drug effects factually -> causal injection should IMPROVE factual NLL. NE: no compensation needed.\n")
ctxA=np.asarray(x[:,:192],dtype=np.float64); uA=np.asarray(x[:,384:],dtype=np.float64); okA=np.asarray(ok)
kI=np.where(mI)[0]; mN=hn_&np.isfinite(rN)&okA; kN=np.where(mN)[0]; FF=[0.5,1.0,2.0,3.0]
t0=time.time(); u0=np.asarray(u_for(int(kI[0]),1.0,1.0),dtype=np.float64).ravel(); print('t1',round(time.time()-t0,4),'err',np.abs(u0-uA[kI[0]]).max().round(5))
UI=np.array([np.asarray(u_for(int(k),f,1.0),dtype=np.float64).ravel() for k in kI for f in FF]); yI=np.array([f*rI[k]-pb[k] for k in kI for f in FF]); cI=np.repeat(kI,4); fI_=np.tile(FF,len(kI))
UN=np.array([np.asarray(u_for(int(k),1.0,f),dtype=np.float64).ravel() for k in kN for f in FF]); yN=np.array([f*rN[k] for k in kN for f in FF]); cN=np.repeat(kN,4); fN_=np.tile(FF,len(kN))
print('aug',round(time.time()-t0,1),UI.shape,UN.shape)
def ridge(X,y,lam):
    mu=X.mean(0); Xc=X-mu; w=np.linalg.solve(Xc.T@Xc+lam*np.eye(X.shape[1]),Xc.T@(y-y.mean())); return w,y.mean()-mu@w
neg=np.where(okA&~hi_&~hn_)[0]; Ug=np.r_[UI,UN,uA[neg]]
tIg=np.r_[np.ones(len(UI)),np.where(hi_[cN],1.,-1.),-np.ones(len(neg))]; tNg=np.r_[np.where(hn_[cI],1.,-1.),np.ones(len(UN)),-np.ones(len(neg))]
GATE={}
for nm,t in [('I',tIg),('N',tNg)]:
    w,b=ridge(Ug,t,1.0); s=Ug@w+b; GATE[nm]=(w,b); print('gate',nm,'pos min',s[t>0].min().round(3),'neg max',s[t<0].max().round(3))
def cvfit(X,y,grp,lams,f1):
    st=np.unique(grp); fold=dict(zip(st,np.random.RandomState(0).randint(0,5,len(st)))); fo=np.array([fold[g] for g in grp]); res={}
    for lam in lams:
        pr=np.zeros_like(y)
        for q in range(5):
            tr_=fo!=q; w,b=ridge(X[tr_],y[tr_],lam); pr[~tr_]=X[~tr_]@w+b
        res[lam]=1-((y-pr)**2).mean()/y.var(); print(' lam',lam,'cvR2',round(res[lam],4),'res sd all',round(np.sqrt(((y-pr)**2).mean()),3),'f=1',round(np.sqrt(((y-pr)[f1]**2).mean()),3))
    return max(res,key=res.get)
XI=np.c_[ctxA[cI],UI]; lI=cvfit(XI,yI,S3a[cI],[0.1,1,10],fI_==1); DEC_I=ridge(XI,yI,lI)
lN=cvfit(UN,yN,S3a[cN],[0.1,1,10],fN_==1); DEC_N=ridge(UN,yN,lN)
np.savez('/work/dec.npz',gIw=GATE['I'][0],gIb=GATE['I'][1],gNw=GATE['N'][0],gNb=GATE['N'][1],dIw=DEC_I[0],dIb=DEC_I[1],dNw=DEC_N[0],dNb=DEC_N[1])
```