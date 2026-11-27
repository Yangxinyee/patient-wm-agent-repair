The rest of the file confirms the readout. It is the value head's predicted mean of finger glucose (or arterial MAP for NE), edited minus factual, at hours after the drug ran. It is averaged over 20 common-random-number paths. In the rollout every drug atom from hour 12 on is scaled, and the model's sampled readings are fed back.

So a value-head edit acts on the readout directly and on the fed-back samples. Dose information reaches the head only through the view. An upstream "clean channel" would mean editing several nonlinear transformer layers that I cannot reproduce exactly offline, so I will stay at the head.

The best quantity to decode for NE is the rate minus its expectation given the MAP history alone. That deviation carries the accumulated titration noise, so it should be far more decodable than the one-step surprise. It is also the right correction under counterfactuals.

Next I estimate E[NE rate | MAP history] with a gradient-boosting regression on MAP-history features across all training stays, and form the deviation D. Then I:

1. regress the factual MAP residuals on D at lags 1 to 3 (expected: near the true effects if the model ignores the rate);
2. check how well with_code decodes D at lag 1 (hoping for R² well above 0.1).

```python
from sklearn.ensemble import HistGradientBoostingRegressor
pt = pt.sort_values(['stay_id','t']).reset_index(drop=True)
g_ = pt.groupby('stay_id')
F_ = pd.DataFrame({'t': pt['t']})
for j in range(0,10):
    F_[f'm{j}'] = g_['Mdec'].shift(j)
for b in (0,1,2):
    F_[f'c{b}'] = (pt['band']==b).astype(float).groupby(pt['stay_id']).cumsum()
low = (pt['Mdec']<65).astype(float)
F_['nlow_ever'] = low.groupby(pt['stay_id']).cumsum()
F_['mmin'] = pt.groupby('stay_id')['Mdec'].cummin()
first_low_t = pt['t'].where(pt['Mdec']<65).groupby(pt['stay_id']).cummin()
F_['since_low'] = pt['t'] - first_low_t
last_low_t = pt['t'].where(pt['Mdec']<65).groupby(pt['stay_id']).ffill()
F_['since_lastlow'] = pt['t'] - last_low_t
sid = pt['stay_id'].values
tr_mask = (sid % 3 != 0)
hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=100)
hgb.fit(F_.values[tr_mask], pt['ne'].values[tr_mask])
Eh = hgb.predict(F_.values)
y_ = pt['ne'].values
r2te = 1-((y_-Eh)[~tr_mask]**2).mean()/y_[~tr_mask].var()
pt['Ene_map'] = Eh; pt['D_ne'] = y_-Eh
print('R2 rate|MAPhist (test)', round(float(r2te),3), 'sd D', round(float(pt['D_ne'][~tr_mask].std()),3), 'sd rate', round(float(y_.std()),3))
Dm = pt.set_index(['stay_id','t'])['D_ne']
for k in (1,2,3):
    ix = pd.MultiIndex.from_arrays([RD2['stay_id'].values, (RD2['hour']-k).values])
    RD2[f'D_ne_{k}'] = Dm.reindex(ix).fillna(0).values
for code in ['map_arterial','map_cuff']:
    print(code, reg(ev==code, ['D_ne_1','D_ne_2','D_ne_3']))
YD = RD2.loc[rm, ['D_ne_1','D_ne_2']].values
print('dec D wc', cvr2(WC0[rm], YD, RD2['stay_id'].values[rm]), 'state', cvr2(XALL[pr_[rm]], YD, RD2['stay_id'].values[rm]))
```