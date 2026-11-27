The tensor list matches the architecture. The dose pathway is `embed.numeric.weight` (192×6: z_value, has_value, z_dose, has_dose, z_rate, has_rate), `embed.bins.dose` and `embed.bins.rate`, then `encode_action`, then the core. The value head reads glucose through `attr_heads.value_bin` plus a transition table keyed on the last bin.

Next I need to understand the records' causal structure:
- whether hours are integers;
- the reading cadence;
- when insulin and norepinephrine records occur, and whether they follow previous readings;
- how doses vary within a stay.

I will print one stay with both drugs and compute summary statistics: the fraction of stays with each drug, the first-dose hour, and whether rates change hour to hour. This decides which estimator of the true effect is possible. If dose assignment has random variation given recent readings (for example, the rate depends on the last glucose plus noise), then regressing the next glucose change on dose while adjusting for the readings that drove the dose identifies the effect.

```python
tr['hour_frac'] = tr.hour - np.floor(tr.hour)
print("fraction non-integer hours:", (tr.hour_frac>1e-9).mean())
print(tr.loc[tr.hour_frac>1e-9].head(10).to_string())
g = tr.groupby('stay_id')
has_ins = g.apply(lambda d: (d.event=='insulin').any())
has_ne = g.apply(lambda d: (d.event=='norepinephrine').any())
print("frac stays insulin", has_ins.mean(), "norepi", has_ne.mean(), "both", (has_ins&has_ne).mean())
sid = has_ins[has_ins & has_ne].index[0]
d = tr[tr.stay_id==sid]
print(d.drop(columns=['hour_frac']).to_string(max_rows=200))
```