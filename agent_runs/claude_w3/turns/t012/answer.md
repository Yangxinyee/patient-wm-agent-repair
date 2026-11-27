Insulin lowers glucose in these records, though modestly over one hour. Each unit/h lowers the next-hour finger reading by about 2 mg/dL (−1.98 ± 0.05 overall; −1.6, −2.0 and −2.2 in the three lowest bands). Within band 1 the raw means fall steadily with dose. The per-unit effect grows slightly with the glucose level, roughly in proportion to it (about −0.01 × G per unit per hour), so the response may be multiplicative rather than additive.

The readout accumulates over 12 hours of doubled doses. What matters is therefore how the effect persists and builds over lags 1–12 while later doses are held as recorded, and whether it is linear in the dose. Before multi-hour estimates, I need the measurement schedule. If a finger reading at t+L is only taken when the last reading exceeded 150, then outcomes are missing in a way that depends on the treatment, and I must handle that. I will tabulate the chance of a finger reading at hour h against the last glucose and its age, and against whether insulin ran in the previous hour.

```python
pn['pol_last'] = np.where(pn.glucose_finger_age.fillna(99) <= pn.glucose_lab_age.fillna(99), pn.glucose_finger_last, pn.glucose_lab_last)
g = pn.groupby('stay_id')
pn['pol_last_prev'] = g.pol_last.shift(1)
pn['gf_present'] = pn.glucose_finger.notna().astype(int)
pn['ins_prev'] = g.insulin.shift(1).fillna(0)
w = pn[(pn.h>=1)].copy()
w['plb'] = pd.cut(w.pol_last_prev, [0,100,120,140,150,160,180,250,400])
print(w.groupby(['plb'], observed=True).agg(n=('gf_present','size'), p_finger=('gf_present','mean'),
      p_ins_prev=('ins_prev', lambda x:(x>0).mean())).round(3).to_string())
# when previous-hour policy glucose <=150: schedule pattern? age since last finger
w2 = w[w.pol_last_prev<=150].copy()
w2['age_prev'] = g.glucose_finger_age.shift(1).reindex(w2.index)
print(w2.groupby('age_prev').gf_present.agg(['size','mean']).head(12).round(3).to_string())
# lab schedule
w['gl_present'] = w.glucose_lab.notna().astype(int)
print("lab presence by hour mod:", w.groupby(w.h % 6).gl_present.mean().round(3).to_dict())
print("lab presence by prev pol band:", w.groupby('plb', observed=True).gl_present.mean().round(3).to_dict())
```