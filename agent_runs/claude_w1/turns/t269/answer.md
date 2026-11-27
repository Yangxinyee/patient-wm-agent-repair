`G3` maps each stay to its record (columns: stay_id, hour, event, value, rate, rate_unit, dose, dose_unit, end_hour), so I can run the experiment. For up to 300 train stays, I'll generate head inputs with the exact replica twice: once with recorded doses and once with insulin doubled from hour 12. On finger rows from hour 13 on I'll measure:
- the lag-1 dose change actually seen;
- the size and consistency across rows of the head-input change;
- the response of the ridge decoder (refit on `V`, λ=100).

If the fresh response agrees with `EDR` (about 0.1 per 1.6 dose change), the head input really carries little dose-caused signal. If it is much larger, the stored `EDR` was stale.

```python
keys3 = list(G3.keys())[:300]
def rows_of(rr, code_='glucose_finger', hmin=13):
    R_ = [r for r in rr if r[0] == code_ and r[2] >= hmin]
    return np.array([r[2] for r in R_]), np.array([r[5:8] for r in R_], float), np.array([r[8] for r in R_])
A_ = V[fm]; mA_, my_ = A_.mean(0), F[fm,3].mean()
wR = np.linalg.solve((A_-mA_).T @ (A_-mA_) + 100*np.eye(192), (A_-mA_).T @ (F[fm,3]-my_))
dXs, dds, hs_ = [], [], []
for s_k in keys3:
    df0 = G3[s_k]; df2 = df0.copy()
    m2_ = (df2['event'] == 'insulin') & (df2['hour'] >= 12)
    df2.loc[m2_, ['rate', 'dose']] *= 2.0
    h_f, L_f, X_f = rows_of(gen_samples(df0, pert=False)); h_e, L_e, X_e = rows_of(gen_samples(df2, pert=False))
    if len(h_f) == 0 or len(h_f) != len(h_e): continue
    dXs.append(X_e - X_f); dds.append(L_e - L_f); hs_.append(h_f)
dXs, dds = np.vstack(dXs), np.vstack(dds)
mu_d = dXs.mean(0) / dds[:,0].mean()
cosr = (dXs @ mu_d) / (np.linalg.norm(dXs, axis=1) * np.linalg.norm(mu_d) + 1e-9)
print("n rows", len(dds), "mean dd lags", np.round(dds.mean(0),3), "mean |dX|", np.round(np.linalg.norm(dXs,axis=1).mean(),3),
      "mean cos to mean dir", np.round(np.nanmean(cosr),3))
print("ridge-dec response mean", np.round((dXs @ wR).mean(),3), "corr w/ dd1", np.round(np.corrcoef(dXs @ wR, dds[:,0])[0,1],3))
print("EDR ins x2 mean |dX|", np.round(np.linalg.norm(EDR[('insulin',2.0)]['dX'],axis=1).mean(),3))
```