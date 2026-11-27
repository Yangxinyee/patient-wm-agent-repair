Good. Everything I need is loaded. Now I'll compute the effective insulin and norepinephrine rate at each of the 5301 bundles from train.csv intervals, and regress each rate on the `out` component of the state (first 192 dims of SX). If a direction in `out` explains rate well, I can replace my value-head rate readers with those directions. That bypasses the fragile core→q channel and uses the signal the original model already propagates.

Outcome that confirms: ridge regression gives R²>0.3 for insulin and a clearly identifiable direction. Then rebuild RR and re-save the value-head edit, keeping the gates and tilt unchanged.

```python
import numpy as np, pandas as pd, os
from sklearn.linear_model import Ridge
stays_set = set(np.unique(SST).tolist())
tr = pd.read_csv('/data/train.csv', usecols=['stay_id','hour','event','rate','end_hour'])
tr_s = tr[tr['stay_id'].isin(stays_set)]
def evs(e): return tr_s[(tr_s['event']==e) & tr_s['rate'].notna()][['stay_id','hour','rate','end_hour']].values
ins_evs = evs('insulin'); ne_evs = evs('norepinephrine')
print('ins events:', len(ins_evs), 'ne events:', len(ne_evs), 'stays:', len(stays_set))
def build(events):
    by_s = {}
    for s,h,r,e in events: by_s.setdefault(int(s),[]).append((h,e,r))
    R = np.zeros(len(SST))
    for i in range(len(SST)):
        s=int(SST[i]); h=SHR[i]
        for h0,e0,r0 in by_s.get(s,[]):
            if h0 <= h < e0: R[i]=r0; break
    return R
RINS = build(ins_evs); RNE = build(ne_evs)
x = np.load('/work/fwd_2/states_original.npz')
OUT_ = x['x'][:,:192].astype(np.float64)
ok = (SHR >= 0)
print('ok n:', int(ok.sum()), 'ins>0:', int((RINS[ok]>0).sum()), 'ne>0:', int((RNE[ok]>0).sum()))
print('ins rate q[.05,.5,.95]:', np.round(np.quantile(RINS[(RINS>0)&ok], [.05,.5,.95]),2))
print('ne rate q[.05,.5,.95]:', np.round(np.quantile(RNE[(RNE>0)&ok], [.05,.5,.95]),2))
R_rate = {}
for name, Y in (('ins',RINS),('ne',RNE)):
    X_=OUT_[ok]; Y_=Y[ok]
    for a in [1,10,100,1000,10000]:
        r=Ridge(alpha=a,fit_intercept=True).fit(X_,Y_); pr=r.predict(X_); r2=r.score(X_,Y_)
        pos=Y_>0; r2p=1-((pr[pos]-Y_[pos])**2).sum()/max(1e-9,((Y_[pos]-Y_[pos].mean())**2).sum())
        print(f'{name} α={a}: R²all={r2:.3f} R²pos={r2p:.3f} |w|={np.linalg.norm(r.coef_):.3g} int={r.intercept_:.4f}')
    R_rate[name] = (r.coef_.copy(), r.intercept_)
# save for next run
import pickle
pickle.dump({'R_rate':R_rate,'RINS':RINS,'RNE':RNE,'OUT_':OUT_,'ok':ok}, open('/work/rate_regs.pkl','wb'))
print('saved /work/rate_regs.pkl')
```