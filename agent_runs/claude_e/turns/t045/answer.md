The saved states file holds, per bundle, the 992-number state (out, the 4 slots of h, the z posterior mean), plus the stay, the bundle time and drug flags.

The question now is whether that state encodes the insulin dose beyond what the glucose reading explains. That within-tier dose variation is the protocol's random multiplier, the part that carries the causal effect. For each insulin bundle I will:
1. Take the recorded rate and the finger reading the protocol used at that hour.
2. Remove the tier means from the dose.
3. Remove the tier and glucose effects from the state.
4. Fit a cross-validated ridge of the dose residual on the state residual.

A within-tier R² well above 0 (say >0.5) means a linear dose readout from the state is feasible. A value near 0 means the state barely carries the dose, which would explain the model's weak response and rule out a pure head-level fix.

```python
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
X=S2['x']; sid=S2['stay_id']; hr=S2['hour']; hi=S2['has_insulin']; hn=S2['has_norepinephrine']
def tier_of(g):
    return np.select([g<150,g<180,g<220,g<300],[0,1,2,3],4)
idx=np.where(hi)[0]; H0=np.floor(hr[idx]+1e-6).astype(int)
dose=np.array([Gtu['ins'].loc[s,h] for s,h in zip(sid[idx],H0)],float)
gf=np.array([Gtu['gf'].loc[s,h] for s,h in zip(sid[idx],H0)],float)
ok_=np.isfinite(dose)&np.isfinite(gf); print("ins bundles",len(idx),"ok",ok_.sum(), "dose>0",(dose>0).sum())
idx,dose,gf=idx[ok_],dose[ok_],gf[ok_]; T=tier_of(gf)
D=np.eye(5)[T]; C=np.column_stack([D, D*(gf[:,None]-200)/50])
def resid(Y,C):
    b=np.linalg.lstsq(C,Y,rcond=None)[0]; return Y-C@b
rd=resid(dose,C); print("dose sd",dose.std().round(3),"within-tier sd",rd.std().round(3))
for nm,sl in [('out',slice(0,192)),('h',slice(192,960)),('z',slice(960,992)),('all',slice(0,992))]:
    Xr=resid(X[idx][:,sl].astype(float),C)
    p=cross_val_predict(RidgeCV(alphas=np.logspace(-2,4,13)),Xr,rd,cv=KFold(5,shuffle=True,random_state=0))
    print(nm,"CV R2 within-tier dose:",round(1-np.mean((rd-p)**2)/np.var(rd),3))
```