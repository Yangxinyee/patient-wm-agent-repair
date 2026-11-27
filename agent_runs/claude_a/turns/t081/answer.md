The head replica is confirmed. My own 4-draw versus 64-draw rmse (2.03 / 0.92) matches the tool's versus 64 (2.06 / 0.90), and my 4-draw NLL of 1.4045 matches the tool's 1.4039. So the offline factual NLL can be trusted for checking the 0.005-nat constraint.

The forward-1 CSV gives the readout per stay-hour: `shift`, `shift_se`, `factual` (the predicted mean under recorded doses) and `reached` (whether the drug had already run before that hour). Forward 4 was a `states` request on an edited model that is no longer in place.

Next I'll read both summaries and work out exactly how the headline numbers are aggregated: pooled over reached stay-hours, or averaged per stay first. That way my offline replica reports the same quantity.

```python
import json
for k_ in [1,4]: print(f"fwd_{k_} summary:", open(f'/work/fwd_{k_}/summary.json').read())
D1f = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
for ed_, g_ in D1f.groupby('edit'):
    rr = g_[g_.reached]
    print(ed_, "rows", len(g_), "reached", len(rr), "stays", rr.stay_id.nunique(), "pooled mean", round(rr['shift'].mean(),4),
          "per-stay mean", round(rr.groupby('stay_id')['shift'].mean().mean(),4), "hours", sorted(g_.hour.unique())[:3], '...', sorted(g_.hour.unique())[-2:])
print(json.load(open('/work/sel1.json')) if len(open('/work/sel1.json').read())<600 else open('/work/sel1.json').read()[:300])
```