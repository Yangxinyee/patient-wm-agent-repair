No cross-effects show up either. Norepinephrine on glucose gives +0.6 ± 0.4 at 1 h and +1.0 ± 0.7 at 6 h, both under 1.5σ, and +0.05 ± 0.52 at 3 h. Insulin on MAP is tightly null (±0.02 per u/h), and glucose dynamics do not depend on MAP. The simulator looks like one where the recorded drugs move nothing measurable. That supports "absent" as the true answer rather than an effect I am failing to see.

To quantify the insulin uncertainty properly, I want to build the evaluation's readout myself: the shift at each "reached" stay-hour is the sum over doses since hour 12 of a lag effect h(k) times the extra dose. I need to know exactly which stay-hours count as reached. The first tuning stay is flagged reached from hour 13 on, so next I will compare the tool's reached flags with the actual drug records for several stays to infer the rule.

```python
f3 = pd.read_csv('/work/fwd_3/dose_edit_original.csv')
for sid in [2020000000, 2020000016, 2020000001]:
    rec = tu[(tu.stay_id==sid)&(tu.event.isin(['insulin','norepinephrine']))&(tu.hour>=10)][['hour','event','rate','end_hour']]
    print(sid, "drug records from h10:"); print(rec.to_string(index=False))
    r = f3[(f3.stay_id==sid)&(f3.edit.isin(['insulin_x2','ne_x2']))][['hour','edit','reached','shift']]
    print(r.pivot(index='hour', columns='edit', values='reached').T.to_string())
    gf = tu[(tu.stay_id==sid)&(tu.event.isin(['glucose_finger','map_arterial']))&(tu.hour>=12)].groupby('event').hour.apply(list)
    print(gf.to_dict())
```