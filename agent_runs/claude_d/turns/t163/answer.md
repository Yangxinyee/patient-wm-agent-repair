The offline value head is now exact. `Hh0` is the post-GELU hidden layer and `A0` its pre-activation. Recomputing from `WC0` reproduces `P0` with zero error, the predicted mean is exactly `pred0` (the sum of `P[1:]` times the bin centres), and `nll0` equals −log P[observed bin]. Class 0 ("no value") has negligible probability.

So I can compute the factual NLL change of any head edit exactly offline, up to the tool's sampling of z. The 400 stays hold 8,214 glucose readings and 11,237 MAP readings.

Next I want to know what the model's factual predictions miss. Two regressions use the exogenous dose shocks from my earlier analysis:
- For glucose readings: the observation, the model prediction and the residual on the insulin dose surprise at lags 1 to 3.
- For MAP readings: the same three on the clean NE instrument at lags 1 to 4.

If the model under-uses dose information, the residual will load on the shocks with roughly "true effect minus model response". For example, glucose lag 1 should then be about −1.5 per u/h. That would mean a centred head correction improves the factual fit rather than costing NLL. If the residual coefficients come out near zero, the model already fits these shocks factually, and the rollout failure lies elsewhere.

```python
import numpy as np, pandas as pd
d_=RD2.copy(); d_['obs']=d_['observed'].astype(float)
print('eps_ins_1 nan%',d_.eps_ins_1.isna().mean().round(3),'nz%',(d_.eps_ins_1.fillna(0)!=0).mean().round(3),'sd',d_.eps_ins_1.std().round(3))
print('NEI nan%',np.isnan(NEI).mean().round(3),'nz%',(np.nan_to_num(NEI)!=0).mean().round(3),'sd',np.nanstd(NEI).round(3),'si ok',si[stays[0]]==0 if 'stays' in globals() else None)
print('gam_clean',np.round(gam_clean,2),'beta_ins',np.round(beta_ins,2))
def ols_(y,X):
    X=np.column_stack([np.ones(len(X)),X]); b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b
    cov=np.linalg.inv(X.T@X)*(r@r/(len(y)-X.shape[1])); return b,np.sqrt(np.diag(cov))
cols=['eps_ins_1','eps_ins_2','eps_ins_3']
g_=d_[d_.event.isin(['glucose_finger','glucose_lab'])].copy()
for c in cols: g_[c]=g_[c].fillna(0)
for yv in ['obs','pred0','res']:
    b,se=ols_(g_[yv].values,g_[cols].values); print('gluc',yv,np.round(b[1:],2),'se',np.round(se[1:],2),len(g_))
mp_=d_[d_.event.isin(['map_arterial','map_cuff'])].copy()
ii_=np.array([si[s] for s in mp_.stay_id]); hh_=mp_.hour.values.astype(int)
Xn_=[]
for k in range(1,5):
    hk=hh_-k; Xn_.append(np.where(hk>=0, np.nan_to_num(NEI[ii_,np.clip(hk,0,25)]), 0.0))
Xn_=np.column_stack(Xn_)
for yv in ['obs','pred0','res']:
    b,se=ols_(mp_[yv].values,Xn_); print('map',yv,np.round(b[1:],2),'se',np.round(se[1:],2),len(mp_))
```