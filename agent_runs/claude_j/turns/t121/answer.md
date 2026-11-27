The training reading table has 4,820 readings (1,435 glucose_finger, 516 glucose_lab, 1,457 map_arterial, 1,412 map_cuff), each with its query vector q. Next I will build the insulin regression targets for each glucose reading at hour t:
- R = Σ_k KI(k)·dose_{t−k}, the true causal dose-history term;
- C = Σ_k KI(k)·c(GF_{t−k}), where c is the expected dose under the sliding scale, estimated from all training readings;
- the deviation T = R − C.

I will then check, with cross-validation grouped by stay, how well a linear readout of q recovers R, C and T. The approach works only if T is recoverable with a reasonable R², because that is the signal the repurposed unit must carry.

```python
print('KI[:4]', np.round(KI[:4], 3), 'KN[:4]', np.round(KN[:4], 3))
gfall = tr[(tr.event == 'glucose_finger')][['stay_id', 'hour', 'value']].copy()
gfall = gfall[np.isclose(gfall.hour % 1, 0)]
insall = tr[tr.event == 'insulin'][['stay_id', 'hour', 'rate']].copy(); insall['j'] = np.floor(insall.hour).astype(int)
gfall['j'] = gfall.hour.astype(int)
mg_ = gfall.merge(insall[['stay_id', 'j', 'rate']], on=['stay_id', 'j'], how='left').fillna({'rate': 0.0})
bands_ = [-1, 150, 180, 220, 260, 300, 1e9]
mg_['band'] = pd.cut(mg_.value, bands_, labels=False)
cband = mg_.groupby('band').rate.mean()
print('E[dose|band]', np.round(cband.values, 3), 'P(given|band)', np.round(mg_.groupby('band').rate.apply(lambda x: (x > 0).mean()).values, 3))
def cexp(g):
    return float(cband.iloc[int(np.searchsorted(bands_, g, side='left') - 1)]) if np.isfinite(g) else 0.0
```