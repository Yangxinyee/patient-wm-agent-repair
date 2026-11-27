The one-hour test for norepinephrine is also null on the main outcome. Holding the previous rate and current MAP fixed, the current rate does not change the next arterial MAP: +0.005 ± 0.09 mmHg per mcg/kg/min, while the rate itself spans 0–1.

The innovations show no autocorrelation (−0.026). Two things need a closer look:
- **Cuff MAP regression:** it gives a negative coefficient (−1.1 ± 0.23). That is suspicious because it relies on hours where cuff readings happen to exist both now and next hour.
- **Small cuff correlation:** the innovation correlates slightly with cuff MAP given arterial MAP (0.0012 per mmHg, significant but tiny). So my policy model, which uses arterial MAP when present and cuff otherwise, is not quite the clinician's trigger.

Next I will take two steps:
1. Determine which MAP reading drives titration when both are present: arterial, cuff, their mean or the minimum. I will compare how well each predicts the band behaviour.
2. With the corrected bands, rebuild the innovations and regress future MAP (arterial and cuff, lags 1–12) on the innovation, controlling for the history. This captures any delayed or cumulative effect, since a shock raises all later rates.

```python
d = r[r.map_arterial.notna() & r.map_cuff.notna()].copy()
d['Mmean'] = (d.map_arterial + d.map_cuff)/2
d['Mmin'] = np.minimum(d.map_arterial, d.map_cuff)
def band_fit(M, y):
    b = mband(M)
    # R^2 of lratio explained by band means
    means = pd.Series(y).groupby(b).transform('mean').values
    return 1 - np.var(y-means)/np.var(y)
for nm in ['map_arterial','map_cuff','Mmean','Mmin']:
    print(nm, round(band_fit(d[nm].values, d.lratio.values),4))
# Look at increase probability around 65 by arterial vs cuff
d['up'] = d.lratio>0
print(d.groupby(pd.cut(d.map_arterial,[50,60,63,64,65,66,67,70,80])).lratio.mean().round(3).to_dict())
print(d.groupby(pd.cut(d.map_cuff,[50,60,63,64,65,66,67,70,80])).lratio.mean().round(3).to_dict())
```