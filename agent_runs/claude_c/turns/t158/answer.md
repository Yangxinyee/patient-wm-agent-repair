The factual files hold one row per reading: stay, hour, event, observed value, predicted mean and NLL. Next I'll join the original and zero-token predictions. I'll measure the mean absolute shift for glucose and MAP, then regress the shift on the rate given in the hour before each reading. That shows how strongly the token, at its current strength, moves predictions per unit rate.

```python
m = fo.merge(fe[['stay_id','hour','event','predicted_mean']], on=['stay_id','hour','event'], suffixes=('_o','_e'))
m['d'] = m.predicted_mean_e - m.predicted_mean_o   # zero-token minus original
tu = pd.read_csv('/data/tuning.csv')
print(tu.columns.tolist()[:15], len(tu))
for ev in ['glucose_finger','map_arterial']:
    x = m[m.event==ev].d
    print(ev, round(x.abs().mean(),3), round(x.mean(),3), round(x.std(),3))
```