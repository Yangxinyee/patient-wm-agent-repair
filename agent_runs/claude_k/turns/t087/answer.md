Forward passes 5–7 were "states" requests on 300 further training stays. The dicts D5, D6 and D7 hold, per reading, the q and with_code vectors, plus code, last bin and record row. D99 holds the same for the first 99 stays. Factual errors from the tool exist only for those 99 stays (forward pass 4).

So I can fit dose probes on about 22,000 readings from 399 stays, but estimate the model's missing dose effect only from the 99 stays.

The probe has two parts:
- The "policy-expected dose" depends on the last glucose or MAP reading. That reading sits in the value head's own register (last-value bin embedding plus z-score), so this part can be read almost exactly.
- The dose itself has to come from the state.

This run joins the with_code vectors to the record-derived doses and shocks at lags 1–3. It then checks with stay-grouped cross-validation:
- how well a ridge probe recovers the raw lag-1 dose and the lag-1/lag-2 shocks;
- how shrunk the probe's response is (the slope of its prediction on the truth).

That slope tells me how much the gain must exceed the factual-optimal one. It also checks that `ug` is the gate vector from run 73.

I expect the raw lag-1 dose to be recovered well (R² ≥ 0.8). The shock R² should be around 0.4, with slope well below 1. If even the raw dose is poorly encoded, a correction read from the state will be noisy, and I will need a different way to reach the dose.

```python
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
DD = [D99, D5, D6, D7]
Rcat = pd.concat([d_['R'] for d_ in DD], ignore_index=True)
WCc = np.vstack([np.asarray(d_['WCb'], dtype=np.float64) for d_ in DD])
codec = np.concatenate([np.asarray(d_['code']) for d_ in DD]).astype(int)
rowsc = np.array([sidx[s] for s in Rcat.stay_id.values])
hrc = Rcat.hour.values.astype(int)
def lagged(Mx, lag):
    j = hrc - lag; ok = (j >= 0) & (j <= 24)
    o = np.zeros(len(hrc)); o[ok] = Mx[rowsc[ok], j[ok]]; return o
print("rows", len(Rcat), "codes", np.unique(codec, return_counts=True), "same as WCall?", WCc.shape, np.abs(WCc - WCall).max() if WCall.shape==WCc.shape else 'n/a')
print("ug gate check corr with pg:", np.corrcoef(WCall @ ug, pg)[0,1] if WCall.shape[0]==pg.shape[0] else None)
grpc = Rcat.stay_id.values
def cv_probe(mask, y, alpha=10.0):
    X_ = WCc[mask]; y_ = y[mask]; g_ = grpc[mask]; pred = np.zeros(len(y_))
    for tr_, te_ in GroupKFold(5).split(X_, y_, g_):
        m_ = Ridge(alpha=alpha).fit(X_[tr_], y_[tr_]); pred[te_] = m_.predict(X_[te_])
    r2 = 1 - ((y_-pred)**2).sum()/((y_-y_.mean())**2).sum()
    slope = np.cov(pred, y_)[0,1]/np.var(y_)
    return r2, slope
mG = np.isin(codec, [6, 10]); mM = np.isin(codec, [5, 7])
for nm_, Mx, msk in [('ins dose', ins_all, mG), ('ins shock', eps_ins, mG), ('ne rate', ne_all, mM), ('ne shock', eps_ne, mM)]:
    for lag in [1, 2, 3]:
        y = lagged(Mx, lag)
        for al in [1.0, 30.0]:
            r2, sl = cv_probe(msk, y, al)
            print(f"{nm_} lag{lag} alpha{al}: sd {y[msk].std():.3f}  CV R2 {r2:.3f}  slope(pred on true) {sl:.3f}")
```