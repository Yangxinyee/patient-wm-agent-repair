Even the full 992-number state encodes the dose deviation poorly: R² is 0.45 at lag 1, 0.21 at lag 2, 0.14 at lag 3 and about 0 from lag 4. The model mostly forgets the dose magnitude within an hour or two. A linear readout from q (R² 0.24) would respond only weakly to a real dose change. Scaling it up to compensate would add noise to factual predictions and likely break the 0.005-nat budget.

Before considering a core edit, I will check whether the information is present nonlinearly. I will compare three decoders of lag-1 and lag-2 deviation and of the combined target: a linear readout of the value head's own hidden layer (192 GELU units, computed for the glucose_finger code), random GELU features of q, and a gradient-boosted model on the full state.

If any of them reaches a much higher R², a rank-1 update of the head's output layer on those features could work. If all stay near 0.3, the information is not in the state and the edit has to create a dose memory upstream.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
rg = RA[gA]; r1 = R1[g1m]
def hid_rows(Xs, R_, nz=16):
    Q = q_from_rows(Xs, torch.as_tensor(R_.prev.values), torch.as_tensor(R_.dt.values, dtype=torch.float64), nz=nz)
    f = reg_feats(torch.as_tensor(R_.has.values), torch.as_tensor(R_.age.values), torch.as_tensor(R_.zl.values))
    _, wc, hid = value_logits(Q, torch.as_tensor(R_.code.values), torch.as_tensor(R_.lvb.values), f)
    return hid.mean(0).numpy()
H = np.vstack([hid_rows(XA, rg), hid_rows(X1, r1)])
Hs = (H-H.mean(0))/(H.std(0)+1e-6)
print('hid linear: lag1', round(cv_r2(Hs, Lg[:,0], grp, 10.)[0],3), ' lag2', round(cv_r2(Hs, Lg[:,1], grp, 10.)[0],3), ' D_t', round(cv_r2(Hs, Dt, grp, 10.)[0],3))
rng2 = np.random.default_rng(0); Wr = rng2.normal(size=(192, 1500))/np.sqrt(192)
Fz = (Fq - Fq.mean(0))/(Fq.std(0)+1e-6); RF = torch.nn.functional.gelu(torch.as_tensor(Fz@Wr)).numpy()
print('random GELU feats of q: lag1', round(cv_r2(RF, Lg[:,0], grp, 30.)[0],3), ' D_t', round(cv_r2(RF, Dt, grp, 30.)[0],3))
Fall = np.vstack([PA_['all'], P1_['all']])
pr = np.zeros(len(Dt)); pr1 = np.zeros(len(Dt))
for tr_, te_ in GroupKFold(5).split(Fall, Dt, grp):
    pr[te_] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05).fit(Fall[tr_], Dt[tr_]).predict(Fall[te_])
    pr1[te_] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05).fit(Fall[tr_], Lg[tr_,0]).predict(Fall[te_])
print('GBM full state: D_t', round(1-np.mean((Dt-pr)**2)/Dt.var(),3), ' lag1', round(1-np.mean((Lg[:,0]-pr1)**2)/Lg[:,0].var(),3))
```