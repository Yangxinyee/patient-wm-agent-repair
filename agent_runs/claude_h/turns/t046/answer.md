The panel is a complete grid of 30,000 stays × 26 hours (−1..24), with norepinephrine rates in `norepinephrine` (NaN when not given). The factual check already shows the dose-invariant edit is free in NLL. The open question is whether norepinephrine truly has a small negative effect after hour 12, or none.

I will redo the norepinephrine analysis in a sharper way:
1. Fit the titration rule flexibly, as rate given current and recent MAP, recent rates and the hour, with cross-fitting. The residual is the random innovation.
2. Remove the predictable part of future arterial MAP using the same history.
3. Regress the remaining MAP on the innovation at each lag (k = 1..12), and the later rates on it for persistence.
4. Deconvolve these into a fixed-dose impulse response.
5. Weight that response by recorded rates at the evaluation-like stay-hours to get an implied ×2 readout, with stay-bootstrap standard errors.

I also split by hour (< 12 vs ≥ 12). Since the innovation is random, the controls only reduce noise; they cannot bias the estimate. I also look at the stored dose-edit output to see which stay-hours the readout uses.

- **Confirms a real effect:** a clearly non-zero readout (beyond about 2.5 se) that is consistent before and after hour 12.
- **Points to absence:** an estimate near zero, or inconsistent across periods.

```python
import numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
t0=time.time()
P=P.sort_index()
stay_ids=P.index.get_level_values(0).unique().to_numpy(); S=len(stay_ids); T=26
A_=lambda c: P[c].to_numpy(dtype=float).reshape(S,T)
GA,GC,NE,INS,GF,GL=[A_(c) for c in ['map_arterial','map_cuff','norepinephrine','insulin','glucose_finger','glucose_lab']]
NE0=np.nan_to_num(NE); INS0=np.nan_to_num(INS)
def lag(X,j):
    Y=np.full_like(X,np.nan)
    if j>0: Y[:,j:]=X[:,:-j]
    elif j==0: Y=X.copy()
    return Y
def cummean(X):
    m=np.where(np.isnan(X),0,X).cumsum(1); c=(~np.isnan(X)).cumsum(1)
    with np.errstate(invalid='ignore',divide='ignore'): return np.where(c>0,m/np.maximum(c,1),np.nan)
hourmat=np.tile(np.arange(-1,25),(S,1)).astype(float)
MAPany=np.where(np.isnan(GA),GC,GA)
feats={'ga0':GA,'ga1':lag(GA,1),'ga2':lag(GA,2),'gc0':GC,'gc1':lag(GC,1),'gc2':lag(GC,2),
       'r1':lag(NE0,1),'r2':lag(NE0,2),'r3':lag(NE0,3),'hr':hourmat,'cmA':cummean(GA),'cmC':cummean(GC),
       'non':lag(NE0>0,1).astype(float).cumsum(1) if False else np.cumsum(lag((NE0>0).astype(float),1),1),
       'cmin':np.fmin.accumulate(np.where(np.isnan(MAPany),np.inf,MAPany),axis=1)}
feats['non']=np.nan_to_num(feats['non']); feats['cmin'][np.isinf(feats['cmin'])]=np.nan
names=list(feats)
# continuing NE rows at time index t (hour t-1): NE given at t-1 and at t
cont=(lag(NE0,1)>0)&(~np.isnan(NE)); cont[:,:3]=False
si,ti=np.nonzero(cont)
X=np.stack([feats[n][si,ti] for n in names],1)
fold=(stay_ids[si]%2).astype(int)
def xfit(y,mask=None):
    pred=np.full(len(y),np.nan)
    ok=~np.isnan(y) if mask is None else mask
    for f in (0,1):
        tr=ok&(fold!=f); te=ok&(fold==f)
        m=HGB(max_iter=200,learning_rate=0.1,max_leaf_nodes=31,min_samples_leaf=100).fit(X[tr],y[tr])
        pred[te]=m.predict(X[te])
    return pred
r=NE[si,ti]; ghat=xfit(r); eps=r-ghat
print('n cont rows',len(r),'eps sd',eps.std().round(4),'rate mean',r.mean().round(3), 'time',round(time.time()-t0,1))
K=12
Lres={}; Rres={}
for k in range(1,K+1):
    tt=ti+k; ok=tt<=25
    y=np.full(len(si),np.nan); y[ok]=GA[si[ok],tt[ok]]
    yt=y-xfit(y); Lres[k]=yt
for j in range(0,K):
    tt=ti+j; ok=tt<=25
    y=np.full(len(si),np.nan); y[ok]=NE0[si[ok],tt[ok]]
    Rres[j]=y-xfit(y)
print('fits done',round(time.time()-t0,1))
def coefs(idx):
    e=eps[idx]; L=[];R=[]
    for k in range(1,K+1):
        y=Lres[k][idx]; ok=~np.isnan(y); L.append((e[ok]*y[ok]).sum()/(e[ok]**2).sum())
    for j in range(K):
        y=Rres[j][idx]; ok=~np.isnan(y); R.append((e[ok]*y[ok]).sum()/(e[ok]**2).sum())
    return np.array(L),np.array(R)
def deconv(L,R):
    e=np.zeros(K)
    for k in range(K):
        e[k]=(L[k]-sum(R[j]*e[k-j] for j in range(1,k+1)))/R[0]
    return e
# evaluation-like weights A(k) on train: rows h in 13..24 (index h+1), GA recorded at h, NE given at some s in [12,h-1]
Aw=np.zeros(K); nrows=0
for h in range(13,25):
    hi=h+1
    ran=(NE0[:, (12+1):hi]>0).any(1) if hi>13 else np.zeros(S,bool)
    rowsm=ran&~np.isnan(GA[:,hi]); nrows+=rowsm.sum()
    for k in range(1,K+1):
        s=h-k
        if s>=12: Aw[k-1]+=NE0[rowsm, s+1].sum()
Aw/=nrows
print('eval-like rows',nrows,'A(k)',Aw.round(3))
L,R=coefs(np.arange(len(si))); e=deconv(L,R)
print('L',L.round(3)); print('R',R.round(3)); print('e',e.round(3)); print('implied x2 readout',(e*Aw).sum().round(3))
# bootstrap by stay
rng=np.random.default_rng(0); us=np.unique(si); pos={}
order=np.argsort(si); sis=si[order]; starts=np.searchsorted(sis,us); ends=np.searchsorted(sis,us,side='right')
bs=[]
for b in range(200):
    pick=rng.integers(0,len(us),len(us))
    idx=np.concatenate([order[starts[p]:ends[p]] for p in pick])
    Lb,Rb=coefs(idx); bs.append(((deconv(Lb,Rb)*Aw).sum(), Lb))
bv=np.array([x[0] for x in bs]); Lb=np.array([x[1] for x in bs])
print('boot readout mean %.3f sd %.3f'%(bv.mean(),bv.std()), 'L se',Lb.std(0).round(3))
for name,msk in [('t<12',(ti-1)<12),('t>=12',(ti-1)>=12)]:
    idx=np.nonzero(msk)[0]; Lm,Rm=coefs(idx); print(name,'n',len(idx),'L',Lm.round(2),'readout',(deconv(Lm,Rm)*Aw).sum().round(3))
print('time',round(time.time()-t0,1))
d1=pd.read_csv('/work/fwd_1/dose_edit_original.csv'); print(d1.head()); print(d1.columns.tolist(), len(d1))
```