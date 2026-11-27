Bin saturation is a minor issue: only 1.9% of hours 13–24 glucose readings in insulin stays lie above the top edge, and about 2.7% of MAP readings lie below the bottom edge.

NE x2 is a bigger problem. Training NE rates are capped at 1.0, so z_rate ≤ 0.92 and the rate bin is ≤ 12. Doubled rates above 0.5 give z up to 2.6 and land in bin 16, which was only ever trained on high insulin doses. The model's response to NE x2 is therefore extrapolation.

The 15 `type_emb` rows match 5 event kinds × 3 atom kinds. So all measurement readings likely share one type, and q is the same for every measurement code at a given time. Code-specific behaviour then only enters through `embed.code[code]` and the last-value register. That suggests gating by code inside the value-head MLP.

To fit a readout on the head's inputs, I first need to reproduce the head locally from saved states and validate it against the tool's factual predicted means. Next I check whether the sel1 states and the fwd_3 factual output cover the same stays, and how the 100 sel1 stays were chosen.

```python
sel1 = json.load(open('/work/sel1.json'))
print(type(sel1), (len(sel1) if hasattr(sel1,'__len__') else ''), str(sel1)[:300])
s1 = sel1 if isinstance(sel1, list) else sel1.get('stays', sel1)
f3_ids = sorted(f3.stay_id.unique())
print('fwd3 stays', len(f3_ids), f3_ids[:3], f3_ids[-3:], ' overlap with sel1:', len(set(f3_ids) & set(s1)))
print('states stays', len(np.unique(z['stay_id'])), ' == sel1?', set(np.unique(z['stay_id']).tolist())==set(s1))
print(json.load(open('/work/fwd_2/summary.json')))
print(json.load(open('/work/fwd_3/summary.json')))
```