The factual pass gives a baseline value NLL of 1.549 per reading over 5,138 readings. Glucose_finger readings (1,726) have NLL 1.50 and mean absolute error 11 mg/dL, so a predictive sd of roughly 14 mg/dL. map_arterial has MAE about 4 mmHg.

Now the key regression. For each glucose_finger reading I'll regress the model's residual (value − predicted mean) on the insulin policy residual EPS at lags 1–4. Because EPS is independent of the history, each coefficient measures directly how much the model under-responds to a dose surprise at that lag in teacher-forced mode. A lag-1 coefficient near −1.5 would match the rollout's hour-13 shortfall.

```python
fa=pd.read_csv('/work/fwd_4/factual_original.csv'); print(fa.columns.tolist(), fa.shape)
fa['res']=fa['value']-fa[[c_ for c_ in fa.columns if 'mean' in c_][0]]
g_=fa[fa.event=='glucose_finger'].copy(); g_['h']=g_.hour.round().astype(int); g_['i']=[si[s] for s in g_.stay_id]
L=[1,2,3,4]
XL=np.column_stack([np.where(g_.h-l>=0, EPS[g_.i, np.clip(g_.h-l,0,25)], 0) for l in L])
import statsmodels.api as sm_ if False else None
from numpy.linalg import lstsq
Xd=np.c_[np.ones(len(g_)),XL]; cf,_,_,_=lstsq(Xd,g_.res.values,rcond=None)
rs=g_.res.values-Xd@cf; cov=np.linalg.inv(Xd.T@Xd)*rs.var()
print('gf resid on EPS lags1-4:',cf[1:].round(2),'se',np.sqrt(np.diag(cov))[1:].round(2),' resid sd',g_.res.std().round(2),' n',len(g_))
```