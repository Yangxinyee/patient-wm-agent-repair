The model's factual predictions do not respond to insulin jitter. The coefficient on j is +0.38 ± 0.32, against −1.62 ± 0.66 in the observed readings. So the original model attributes the post-insulin fall to the glucose level and the presence of insulin, not to the dose. That explains why its x2 and x1/2 answers are near zero.

The edit must therefore add sensitivity to the dose relative to the band's expected dose: Δ ≈ θ·(dose − μ(band)). On factual records this term is only the jitter part, which is small and correct. Under x2 it becomes the full doubling.

For a head-level edit, the state seen by the heads must linearly encode the recent insulin rate, and ideally its deviation from the band. Next I'll take the fwd_3 states at insulin bundles and run cross-validated ridge regressions of three targets: the rate just started, its band base, and the deviation j.
- High R² for the rate, and for j or base, means a readout direction exists that I can wire into the glucose value head.
- Low R² means the dose signal has to be injected upstream, through the action encoder.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
S3 = np.load('/work/fwd_3/states_original.npz')
X3, sid3, hr3, hi3 = S3['x'], S3['stay_id'], S3['hour'], S3['has_insulin']
st_ins = tu_[tu_.event=='insulin'].copy()
rate_at = {(s_,round(h_,3)):r_ for s_,h_,r_ in zip(st_ins.stay_id, st_ins.hour, st_ins.rate)}
rows = np.where(hi3>0)[0]
rt = np.array([rate_at.get((sid3[i], round(hr3[i],3)), 0.0) for i in rows])
G0 = np.array([pol.get((sid3[i], int(np.floor(hr3[i]+1e-6))), np.nan) for i in rows])
okr = (rt>0)&np.isfinite(G0)
bs = np.select([G0<=180,G0<=220,G0<=260,G0<=300],[1,2,3,4],6).astype(float)
jj = rt-1.056*bs
print('ins bundles',len(rows),'with start',okr.sum(), 'hour frac', np.round(np.unique(np.round(hr3[rows]%1,3))[:5],3))
for nm_, y_ in [('rate',rt[okr]),('base',bs[okr]),('jdev',jj[okr])]:
    for part,sl in [('out',slice(0,192)),('h',slice(192,960)),('all',slice(0,992))]:
        Xp = X3[rows[okr]][:,sl]
        pr_ = cross_val_predict(RidgeCV(alphas=np.logspace(-2,3,12)), Xp, y_, cv=5)
        print(nm_, part, 'R2 %.3f'%(1-np.var(y_-pr_)/np.var(y_)))
```