The residual regressions show the model's factual predictions miss part of the dose effects.

- **Glucose:** the residual loads −1.28, −1.31 and −0.45 (se 0.3) on the insulin dose surprise at lags 1 to 3. The model's own prediction even moves the wrong way.
- **MAP:** the residual loads +2.93 and +2.67 (se 0.8) on the NE shock at lags 1 and 2, and about 0 at lags 3 and 4. So the model under-reacts to recent NE changes and handles older ones once the readings reflect them.
- **Fit stakes:** these effects are small next to the residual spread (15 mg/dL glucose, 5.2 mmHg MAP), so correcting them barely moves the factual NLL. That explains why training never learned them, and it leaves room under the 0.005-nat constraint.
- **Mechanism:** this fits the sequential picture. If the one-step predictions respond correctly to recent dose changes, the drawn readings carry the effect forward in the rollout.

A head-level edit can only use what is linearly present in the head input `WC0` (or the hidden layer). So next I measure the cross-validated decodability of these targets, with folds grouped by stay:
- insulin dose surprise at lags 1 to 3 and the lag-1 insulin rate (glucose rows);
- NE shock at lags 1 and 2, the lag-1 NE rate and the recent NE rate change (MAP rows).

Lag 1 should be well decodable, with R² about 0.5 or more, and lags 2 and beyond weak. If so, the edit will rest on a lag-1 decoder plus a scale fitted in the rollout. If lag 2 is also decodable, I will add it.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
def cvr2_(X,y,groups,alphas=(1,10,100,1000)):
    X=(X-X.mean(0))/(X.std(0)+1e-9); best=(-9,None)
    for a in alphas:
        pr=np.zeros_like(y)
        for tr_,te_ in GroupKFold(5).split(X,y,groups):
            m_=Ridge(alpha=a).fit(X[tr_],y[tr_]); pr[te_]=m_.predict(X[te_])
        r2=1-((y-pr)**2).sum()/((y-y.mean())**2).sum()
        if r2>best[0]: best=(r2,a)
    return round(best[0],3),best[1]
gm_=RD2.event.isin(['glucose_finger','glucose_lab']).values; mm_=~gm_
grp_s=RD2.stay_id.values
out_={}
for c in ['eps_ins_1','eps_ins_2','eps_ins_3','ins_1']:
    y=RD2[c].fillna(0).values[gm_]
    out_['G:'+c]=(cvr2_(WC0[gm_],y,grp_s[gm_]), cvr2_(Hh0[gm_],y,grp_s[gm_]))
ne1_=RD2['ne_1'].fillna(0).values; ne2_=RD2['ne_2'].fillna(0).values
tg_={'NEI1':Xn_[:,0],'NEI2':Xn_[:,1],'ne_1':ne1_[mm_],'dne12':(ne1_-ne2_)[mm_]}
for c,y in tg_.items():
    out_['M:'+c]=(cvr2_(WC0[mm_],y,grp_s[mm_]), cvr2_(Hh0[mm_],y,grp_s[mm_]))
for k,v in out_.items(): print(k,'WC',v[0],'HID',v[1])
```