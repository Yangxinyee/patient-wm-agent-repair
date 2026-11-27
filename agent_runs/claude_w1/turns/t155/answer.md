The tool's factual file holds, for each reading, the observed value and the predicted mean, so I can compute exact residuals of the original model. The policy facts in my notebook raise one central question. Have the model's predictions already absorbed the part of the dose that follows from the readings (the policy part)? If so, an edit that amplifies the dose response must subtract it again. If the model under-responds to the dose level as a whole, plain amplification is safe for factual fit. Test: on setA, regress the original model's residuals on lagged dose innovations (dose minus policy expectation) and on lagged policy expectations, with both policies estimated non-parametrically from the training records.
- Coefficients on the policy part near zero mean the part is absorbed and compensation is needed.
- Coefficients on the policy part similar to those on the innovations mean plain amplification is fine.

```python
import numpy as np, pandas as pd, json
tr = pd.read_csv('/data/train.csv'); tu = pd.read_csv('/data/tuning.csv'); setA = json.load(open('/work/setA.json'))
def hourly(df):
    ids = np.sort(df.stay_id.unique()); idx = pd.Series(np.arange(len(ids)), index=ids); n=len(ids); H=26
    o = {k: np.full((n,H), np.nan) for k in ['gf','gl','ma','mc']}; o['ins']=np.zeros((n,H)); o['ne']=np.zeros((n,H))
    for ev,k in [('glucose_finger','gf'),('glucose_lab','gl'),('map_arterial','ma'),('map_cuff','mc')]:
        s=df[df.event==ev]; hh=np.round(s.hour.values).astype(int)+1; ok=(hh>=0)&(hh<H)
        o[k][idx[s.stay_id.values].values[ok], hh[ok]] = s.value.values[ok]
    for ev,k in [('insulin','ins'),('norepinephrine','ne')]:
        s=df[df.event==ev]; hh=np.floor(s.hour.values).astype(int)+1; ok=(hh>=0)&(hh<H)
        o[k][idx[s.stay_id.values].values[ok], hh[ok]] = s.rate.values[ok]
    o['G']=np.where(np.isnan(o['gf']),o['gl'],o['gf']); o['P']=np.where(np.isnan(o['ma']),o['mc'],o['ma'])
    return ids, idx, o
idsT, idxT, OT = hourly(tr); idsU, idxU, OU = hourly(tu)
pos = OT['ne'][OT['ne']>0]; redges = np.quantile(np.log(pos), np.linspace(0,1,13)[1:-1])
def keys(O):
    G=O['G'][:,1:]; P=O['P'][:,1:]; dpi=O['ins'][:,:-1]; dpn=O['ne'][:,:-1]
    gk = np.where(np.isnan(G), 77+np.clip(np.round(dpi*2),0,20).astype(int), np.clip(((np.nan_to_num(G)-100)//4).astype(int),-1,75)+1) + 100*(dpi>0)
    rb = np.where(dpn>0, 1+np.digitize(np.log(np.maximum(dpn,1e-6)), redges), 0)
    pk = rb*40 + np.where(np.isnan(P), 38, np.clip(((np.nan_to_num(P)-40)//2).astype(int),-1,36)+1)
    return gk.astype(int), pk.astype(int)
gkT, pkT = keys(OT)
tabI = pd.Series(OT['ins'][:,1:].ravel()).groupby(gkT.ravel()).mean()
tabN = pd.Series(OT['ne'][:,1:].ravel()).groupby(pkT.ravel()).mean()
def pis(O):
    gk, pk = keys(O)
    pI = np.zeros_like(O['ins']); pN = np.zeros_like(O['ne'])
    pI[:,1:] = tabI.reindex(gk.ravel()).fillna(0).values.reshape(gk.shape); pN[:,1:] = tabN.reindex(pk.ravel()).fillna(0).values.reshape(pk.shape)
    return pI, pN
pIU, pNU = pis(OU); nuIU = OU['ins']-pIU; nuNU = OU['ne']-pNU
pIT, pNT = pis(OT); print('train innov sd ins %.3f ne %.3f | pi sd ins %.3f ne %.3f' % (np.std((OT['ins']-pIT)[:,1:]), np.std((OT['ne']-pNT)[:,1:]), np.std(pIT[:,1:]), np.std(pNT[:,1:])))
def resreg(df, ev, nu, pi, lags=3):
    s = df[(df.event==ev)&(df.hour>=lags)].copy(); i = idxU[s.stay_id.values].values; c = np.round(s.hour.values).astype(int)+1
    X = np.column_stack([nu[i,c-k] for k in range(1,lags+1)] + [pi[i,c-k] for k in range(1,lags+1)])
    keep = np.abs(X).sum(1)>0; y = (s.observed-s.predicted_mean).values[keep]; X = np.column_stack([np.ones(keep.sum()), X[keep]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); e = y-X@b; se = np.sqrt(np.diag(np.linalg.inv(X.T@X))*e.var())
    print(ev, 'n', keep.sum(), 'nu lags', np.round(b[1:lags+1],2), 'se', np.round(se[1:lags+1],2), '| pi lags', np.round(b[lags+1:],2), 'se', np.round(se[lags+1:],2))
resreg(F2df, 'glucose_finger', nuIU, pIU); resreg(F2df, 'map_arterial', nuNU, pNU)
```