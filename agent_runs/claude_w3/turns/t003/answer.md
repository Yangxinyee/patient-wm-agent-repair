The records are hourly. Measurements fall on whole hours, and each infusion record starts 5 minutes past the hour (x.0833) and runs for one hour. Insulin rates run from 0.5 to 20.5 units/h, apparently in 0.5 steps. Norepinephrine dose equals rate × 4.8 throughout, which implies one fixed body weight. On average a stay has about 9 insulin records and 8 norepinephrine records.

Next I want to see what drives dosing. I will print one full stay that has both drugs, check how many stays receive each drug and when, and relate each insulin rate to the most recent glucose reading (and each norepinephrine rate to the latest MAP) before it. If the rate is a fixed function of the last reading, the only causal leverage would come from the noise around that function or from rules that change with time. If dose varies a lot at a fixed last reading, that variation can be used to estimate the effect, as long as nothing hidden drives it.

```python
pd.set_option('display.width', 250)
ins = tr[tr.event=='insulin']; ne = tr[tr.event=='norepinephrine']
print("stays with insulin:", ins.stay_id.nunique(), " with norepi:", ne.stay_id.nunique())
print("insulin rate value counts (top 25):", ins.rate.round(3).value_counts().head(25).to_dict())
print("norepi rate quantiles:", ne.rate.quantile([0,.05,.1,.25,.5,.75,.9,.95,1]).round(3).to_dict())
print("insulin start-hour fractional parts:", (ins.hour % 1).round(4).value_counts().head(5).to_dict())
print("meas hour fractional parts:", (tr[tr.event.isin(['glucose_finger','map_arterial'])].hour % 1).round(4).value_counts().head(5).to_dict())
# a stay with both drugs
both = set(ins.stay_id) & set(ne.stay_id)
sid = sorted(both)[0]
print(tr[tr.stay_id==sid].to_string())
```