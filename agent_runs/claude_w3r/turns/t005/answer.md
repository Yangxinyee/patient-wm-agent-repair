Several things are now clear. Readings fall on integer hours from -1 to 24. Each infusion record starts 5 minutes after the hour and lasts exactly 1 hour. Every stay has the same length (25.17 h). Weight is fixed at 80 kg, so dose is proportional to rate for both drugs, and dose and rate carry the same information. Insulin rates come in 0.5 steps. Norepinephrine is continuous and capped at 1.0 (13% of records sit at the cap). An infusion started at t+5min can therefore affect readings from hour t+1 on.

Next I will build an hourly panel (stay × hour: each reading and each drug rate) to study the dosing policy, i.e. how rate depends on the latest readings and how much it varies beyond them. Exogenous variation in dosing is what will let me identify the true effects. If rate turns out to be a near-deterministic function of the last glucose (or MAP), I will look for the residual variation. If it is noisy, I can use regression adjustment.

```python
tr['h'] = np.floor(tr.hour).astype(int)
evs = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
P = {}
for ev in evs:
    s = tr[tr.event==ev]
    P[ev] = s.pivot_table(index='stay_id', columns='h', values='value', aggfunc='mean')
for ev in ['insulin','norepinephrine']:
    s = tr[tr.event==ev]
    P[ev] = s.pivot_table(index='stay_id', columns='h', values='rate', aggfunc='sum')
stays = np.sort(tr.stay_id.unique())
hours = np.arange(-1,25)
A = {k: v.reindex(index=stays, columns=hours).values for k,v in P.items()}  # NaN = absent
import pickle; pickle.dump((stays,hours,A), open('/work/panel.pkl','wb'))
for k,v in A.items():
    print(k, 'presence by hour:', np.round(np.mean(~np.isnan(v),axis=0),2))
# multiple records per hour?
print('max drug records per stay-hour', tr[tr.event=='insulin'].groupby(['stay_id','h']).size().max(),
      tr[tr.event=='norepinephrine'].groupby(['stay_id','h']).size().max())
```