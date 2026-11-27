The insulin effect grows with glucose level, roughly in proportion to it. The 3-hour mean effect per U/h is:

| glucose band (mg/dL) | effect per U/h |
|---|---|
| 150–170 | −1.43 |
| 170–190 | −1.81 |
| 190–210 | −2.12 |
| 210–240 | −2.15 |
| >240 | −2.49 |

That is about −0.0097 × G per U/h in every band. This looks like a "minimal-model" mechanism: insulin builds up a decaying action compartment X, glucose disappears at a rate proportional to X × G, and G mean-reverts to a wandering baseline. The rising effect across rate bins is explained by higher rates going with higher glucose. Dosing noise is multiplicative (residual SD grows with rate), and there is no sign of a saturating dose response.

Next I need the original model's own answers. I'll pick tuning stays where insulin and/or NE run after hour 12, so a single dose_edit pass on the original model is informative for both drugs, and compare against my record-based estimates for the same stays. First I'll check the tuning set's composition and choose the stays.

```python
PT = PT.sort_index()
stt = PT.groupby(level=0).agg(ins_after=('insulin', lambda x: (x.loc[(slice(None), slice(12,24))]>0).sum()),
                               ne_after=('norepinephrine', lambda x: (x.loc[(slice(None), slice(12,24))]>0).sum()),
                               art=('map_arterial', lambda x: x.notna().sum()))
print((stt.ins_after>0).sum(), (stt.ne_after>0).sum(), ((stt.ins_after>0)&(stt.ne_after>0)).sum(), ((stt.ne_after>0)&(stt.art>0)).sum())
both_ = stt[(stt.ins_after>0)&(stt.ne_after>0)].index.tolist()
ins_only = stt[(stt.ins_after>0)&(stt.ne_after==0)].index.tolist()
ne_only = stt[(stt.ins_after==0)&(stt.ne_after>0)].index.tolist()
neither = stt[(stt.ins_after==0)&(stt.ne_after==0)].index.tolist()
print(len(both_), len(ins_only), len(ne_only), len(neither))
batch1 = both_[:50] + ins_only[:25] + ne_only[:25]
print(len(batch1)); json.dump({'batch1': [int(x) for x in batch1]}, open('/work/batches.json','w'))
print(json.dumps([int(x) for x in batch1]))
```