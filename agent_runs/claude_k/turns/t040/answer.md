Each hour has a readings bundle at the whole hour and an action bundle at hour + 5 min. The action bundle holds the new insulin/norepinephrine start atoms plus the END atoms of the previous hour's infusions (e.g. 4 atoms when both drugs continue). The state after the hour-t action bundle is what the model uses to predict the hour t+1 readings.

Next I'll align each action bundle with the recorded doses: the current insulin and norepinephrine rates and the rates 1–3 hours earlier. I'll fit cross-validated ridge regressions, grouped by stay, from (a) out only and (b) the full 992-d state to each dose lag, then check calibration by dose level. High out-of-sample R² with predictions tracking the true dose even at the largest rates would mean the state carries a linear dose-history signal for the correction to read. Saturation at high doses or low R² for older lags would limit that approach.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
X = S2['x']; st['t'] = np.floor(st.hour+1e-6).astype(int); st['frac'] = (st.hour - st.t).round(4)
sidx = {s:i for i,s in enumerate(ids)}
act = st[st.frac==0.0833].copy()
ii = np.array([sidx[s] for s in act.stay]); tt = act.t.values
for l in range(4):
    act[f'I{l}'] = [ins_all[i,t-l] if t-l>=0 else 0 for i,t in zip(ii,tt)]
    act[f'N{l}'] = [ne_all[i,t-l] if t-l>=0 else 0 for i,t in zip(ii,tt)]
Xa = X[act.index.values]
def cv_r2(Xm, y, groups, alpha=1.0):
    pred = np.zeros(len(y))
    for a,b in GroupKFold(5).split(Xm, y, groups):
        m = Ridge(alpha=alpha).fit(Xm[a], y[a]); pred[b] = m.predict(Xm[b])
    return 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum(), pred
res_dec = {}
for tgt in ['I0','I1','I2','I3','N0','N1','N2','N3']:
    y = act[tgt].values
    r_out, _ = cv_r2(Xa[:, :192], y, act.stay.values)
    r_all, pred = cv_r2(Xa, y, act.stay.values)
    res_dec[tgt] = pred
    print(tgt, "R2 out:", round(r_out,3), " R2 full:", round(r_all,3))
act['I0hat'] = res_dec['I0']; act['N0hat']=res_dec['N0']
print(act.groupby(act.I0.clip(0,8)).I0hat.agg(['mean','size']).round(2).T)
print(act.groupby((act.N0*10).round().clip(0,10)/10).N0hat.agg(['mean','size']).round(2).T)
```