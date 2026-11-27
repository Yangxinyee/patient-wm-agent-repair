The states file has 5,116 rows on 100 tuning stays: `out`, the 4 slots of h, and the z posterior, one row per bundle. The records give each infusion's start time (hour + 5 min), rate and dose, and the start bundles also hold the END of the previous infusion. I can now test whether recent drug history is linearly readable from the state. I'll use only `out` and the h slots, because the heads use a prior sample of z, not the posterior. If ridge regressions on these 960 numbers predict lagged insulin and norepinephrine rates well out of sample, a head-level edit can work. If R² falls to near zero after one lag, the head cannot see the history.

```python
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
x = S['x']; sid_ = S['stay_id']; hr_ = S['hour']
def hourly(ev):
    d = tu[tu.event==ev][['stay_id','hour','rate']].copy()
    d['h'] = np.round(d.hour-0.0833).astype(int)
    return {(a,b):r for a,b,r in zip(d.stay_id,d.h,d.rate)}
HI = hourly('insulin'); HN = hourly('norepinephrine')
base_h = np.floor(hr_ - 0.08 + 1e-6).astype(int)   # hour of the most recent possible infusion start
def lagged(H, K=5):
    return np.array([[H.get((s,b-k),0.0) for k in range(K)] for s,b in zip(sid_,base_h)])
YI = lagged(HI); YN = lagged(HN)
keep = hr_ >= 1
F_all = x[:, :960]; F_small = np.hstack([x[:, :192], x[:,192:960].reshape(-1,4,192).mean(1)])
def cvr2(F, Y, alpha=10.0):
    F = F[keep]; Y = Y[keep]; grp = sid_[keep]
    pred = np.zeros_like(Y)
    for tr_i, te_i in GroupKFold(5).split(F, Y, grp):
        mu = F[tr_i].mean(0); sd = F[tr_i].std(0)+1e-6
        m_ = Ridge(alpha=alpha).fit((F[tr_i]-mu)/sd, Y[tr_i]); pred[te_i] = m_.predict((F[te_i]-mu)/sd)
    return 1 - ((Y-pred)**2).sum(0)/((Y-Y.mean(0))**2).sum(0)
for nm_, F in [('all960',F_all),('out+meanh',F_small)]:
    for a_ in [10, 100, 1000]:
        print(nm_, a_, 'INS', np.round(cvr2(F,YI,a_),3), 'NE', np.round(cvr2(F,YN,a_),3))
```