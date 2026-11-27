The insulin probe has reached its limit. Its held-out R² is 0.51 at lag 1 and 0.38 for the combined target, the same at 399 stays as at 299. More state passes will not improve it.

That is still workable. If the correction is scaled by 1/R² to give the right response to a changed dose, its added noise roughly cancels the error it removes. I estimate the factual cost at about +0.001 nats overall, within the limit.

Norepinephrine is the problem. The readings-only residual rate (sd 0.146) is almost invisible to a linear probe (R² 0.05). At that quality the scaled correction would add about 6–19 times more noise than it removes, costing far more than 0.005 nats.

Before deciding how to handle norepinephrine, I will check whether the state encodes its dose in a different form. This run probes three alternatives:
- the raw rate;
- the log rate and its readings-only residual;
- indicators of the rate bins.

Readings preceded by an infusion bundle are tested separately, along with the insulin raw dose for comparison. If one form is well decoded (R² ≳ 0.5), I can build the correction on it.

```python
logne = np.log(ne_all + 0.02)
gbL1 = HGB(max_iter=300, learning_rate=0.08).fit(Xne[tr_m], logne.reshape(-1)[tr_m]); gbL2 = HGB(max_iter=300, learning_rate=0.08).fit(Xne[~tr_m], logne.reshape(-1)[~tr_m])
nu_log = logne - np.where(tr_m, gbL2.predict(Xne), gbL1.predict(Xne)).reshape(30000, 25)
ne_edges = np.array(NUM[('norepinephrine', 'rate')]['fine_bin_edges'])
print('ne raw d1', probe_r2(DS, MP, ne_all, np.array([1, 0, 0])))
print('ne log d1', probe_r2(DS, MP, logne, np.array([1, 0, 0])))
print('ne log resid nu1', probe_r2(DS, MP, nu_log, np.array([1, 0, 0])))
print('ins raw d1', probe_r2(DS, GL, ins_all, np.array([1, 0, 0])))
def probe_r2_mask(Ds, evs_, E_, combo, need_prev_drug=True):
    Qs, ys, gs = [], [], []
    for D_ in Ds:
        R_ = D_['R']; m_ = R_.event.isin(evs_).values & (R_.th.values >= 1)
        if need_prev_drug: m_ &= (ne_all[R_.ri.values, np.clip(R_.th.values - 1, 0, 24)] > 0)
        Qs.append(D_['Q'][m_]); ys.append(lags(D_, E_)[m_] @ combo); gs.append(R_.stay_id.values[m_])
    Q_, y_, g_ = np.vstack(Qs), np.concatenate(ys), np.concatenate(gs)
    yp = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), Q_, y_, groups=g_, cv=GroupKFold(5))
    return round(1 - np.mean((y_ - yp) ** 2) / np.var(y_), 3), len(y_)
print('ne (on at t-1) raw d1', probe_r2_mask(DS, MP, ne_all, np.array([1, 0, 0])), 'log d1', probe_r2_mask(DS, MP, logne, np.array([1, 0, 0])), 'nu_log1', probe_r2_mask(DS, MP, nu_log, np.array([1, 0, 0])), 'eps1', probe_r2_mask(DS, MP, eps_ne, np.array([1, 0, 0])))
```