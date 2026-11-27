The dose-invariance edit leaves the factual fit essentially untouched: value NLL per reading 1.49270 → 1.49271 (+0.00001 nats), with every event type unchanged to the fourth decimal. Removing dose size costs nothing for factual prediction, which fits the records: doses add no information about future readings beyond the readings the dose was based on.

Before I lock in readouts of exactly 0, I want to resolve one open number. The norepinephrine DML gave −0.33 ± 0.08 mmHg per mcg/kg/min at each of lags 1–3. If that were real and accumulated over hours of doubled infusion, the true norepinephrine x2 readout could be around −0.5 to −1 mmHg rather than 0. The other design disagreed (+0.09 at lag 1, −0.58 at lag 6), so the titration rule is not fully captured.

Next I will characterise the titration rule exactly: how often the rate is unchanged, the size of steps, and their dependence on MAP, the previous rate and the rate two hours back. With the exact rule I can isolate its truly random part.

```python
H3 = H.copy(); H3['rate']=H3['ne']
c = H3[(H3['rate']>0) & (H3['ne_m1']>0)].copy()
c['ratio'] = (c['rate']/c['ne_m1'])
c['same'] = np.isclose(c['rate'], c['ne_m1'])
c['map_h'] = c['map_arterial'].fillna(c['map_cuff'])
c['mb'] = (c['map_h']//5*5)
c['prev_step'] = np.sign((c['ne_m1']-c['ne_m2']).round(4))
print(c.groupby('mb', dropna=False).agg(n=('same','size'), p_same=('same','mean'), med_ratio=('ratio','median'), up=('ratio', lambda x:(x>1.0001).mean()), down=('ratio', lambda x:(x<0.9999).mean())).round(3).to_string())
print(c.groupby(['prev_step']).agg(n=('same','size'), p_same=('same','mean'), up=('ratio', lambda x:(x>1.0001).mean()), down=('ratio', lambda x:(x<0.9999).mean())).round(3))
# examples of rate sequences
ex = H3.loc[H3.index.get_level_values('stay_id')==c.index.get_level_values('stay_id')[0], ['map_arterial','map_cuff','rate']]
print(ex.T.round(2).to_string())
```