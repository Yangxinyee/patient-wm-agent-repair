More data helps the read-out. Lag-1 insulin deviation on glucose-finger rows decodes with cross-validated R² of 0.34 from 1 file, 0.43 from 3 and 0.44 from 5. It is levelling off around 0.45, much better than the 0.22 from one batch's design rows.

The one-step coefficients on these 500 stays look odd for both truth and model: lags 2–5 come out strongly positive (truth [−2.08, +0.58, +1.61, +2.11, +1.41, +0.31]). The likely cause is my own stay selection. I picked stays with at least 3 insulin hours after hour 12, which selects on later glucose staying high. Earlier insulin affects later glucose, so this selection induces a spurious positive link between earlier doses and later readings.

I'll take two lessons from this:
- the model's own coefficients must come from unselected stays: the random fwd_6 batch and the consecutive tuning stays;
- the read-out fit is mostly unaffected, because it is a mechanical state-to-dose map.

Lag 1 is robust in every sample: the model's −0.4 to −0.7 against the true −2.0.

Next is the norepinephrine side. The protocol-expected rate depends on the previous rate and the MAP band, which I'll estimate empirically from all train stays. From that I'll compute norepinephrine deviations, then check how well the lag-1 deviation decodes from `with_code` on arterial and cuff MAP rows.

If its R² is comparable to insulin's (about 0.4), the same two-unit correction can serve MAP. If it is near zero, the norepinephrine fix needs another route.

```python
ne = Gtr['ne'].fillna(0.0); ma_ = Gtr['ma']; mc_ = Gtr['mc']
mapv = ma_.where(ma_.notna(), mc_)
def band_of(v):
    return np.where(np.isnan(v), 3, np.where(v < 65, 0, np.where(v <= 75, 1, 2)))
hrs_ne = [c for c in ne.columns if c-1 in ne.columns]
rows_ = []
for h in hrs_ne:
    rows_.append(pd.DataFrame({'stay': ne.index.values, 'h': h, 'r': ne[h].values, 'rp': ne[h-1].values, 'band': band_of(mapv[h].values.astype(float))}))
NEd = pd.concat(rows_, ignore_index=True)
pos = NEd.rp > 0
qb = np.quantile(NEd.rp[pos], np.linspace(0, 1, 13)[1:-1])
NEd['rbin'] = np.where(pos, 1 + np.searchsorted(qb, NEd.rp.values), 0)
cell = NEd.groupby(['rbin','band']).r.transform('mean')
NEd['dev'] = NEd.r - cell
print('NE dev sd (rows with r>0 or rp>0):', NEd.dev[(NEd.r>0)|(NEd.rp>0)].std(), ' cells', NEd.groupby(['rbin','band']).ngroups)
DEVNE = NEd.pivot(index='stay', columns='h', values='dev')
for k in [1,2,3]:
    RA_[f'nedev{k}'] = [DEVNE.loc[s, h-k] if (h-k) in DEVNE.columns else 0.0 for s, h in zip(RA_.stay.values, hr)]
    RA_[f'nedev{k}'] = RA_[f'nedev{k}'].fillna(0.0)
for ev in ['map_arterial', 'map_cuff']:
    mm = (RA_.event == ev).values
    Rm, Wm = RA_[mm].reset_index(drop=True), WA_[mm]
    F_ = np.c_[Wm, np.eye(17)[Rm.lvb.values.astype(int)]]
    for k in [1, 2]:
        y = Rm[f'nedev{k}'].values
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2,4,14)), F_, y, cv=GroupKFold(5), groups=Rm.stay.values)
        sl = np.polyfit(y, pred, 1)[0]
        print(f'{ev} nedev{k}: rows {len(y)} sd {y.std():.4f}  CV R2 {1-np.mean((y-pred)**2)/y.var():.3f}  gain {sl:.3f}')
# insulin readout gain (slope of CV prediction on dev1) for glucose_finger
y = Rg.dev1.values; F_ = np.c_[Wg, lvb_oh_g]
predg = cross_val_predict(RidgeCV(alphas=np.logspace(-2,4,14)), F_, y, cv=GroupKFold(5), groups=Rg.stay.values)
print('gf dev1 gain', np.polyfit(y, predg, 1)[0], 'R2', 1-np.mean((y-predg)**2)/y.var())
```