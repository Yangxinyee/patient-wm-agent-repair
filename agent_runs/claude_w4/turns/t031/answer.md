## Step 30: first forward pass, the original model's dose answers

**What the last result showed.** In the 400 tuning stays:
- 231 have insulin at or after hour 12;
- 178 have norepinephrine at or after hour 12;
- 102 have both;
- 111 have norepinephrine and an arterial line, which the arterial-MAP readout needs.

**What I will do next.** Run the evaluation's own rollout (dose_edit) with the original model on 100 of the 102 tuning stays that have both drugs from hour 12. This shows:
- what the model currently predicts for all four edits;
- the output format, including which stay-hours enter the readout.

I will compare that with my estimate of the true effect on the same stays: the insulin kernel (cumulative about −10 per sustained unit/h) and the norepinephrine kernel (about +10 per sustained mcg/kg/min on arterial MAP).

**What would count as evidence.**
- I expect the true insulin ×2 readout to be strongly negative, on the order of −10 mg/dL for typical doses of about 2 units/h.
- If the model predicts something near zero or positive, it has learned the confounded association, and the edit has a lot to do.

```forward
{"type": "dose_edit", "model": "original", "split": "tuning", "stays": [1010043201, 1010043207]}
```

Wait — I need the actual stay IDs. Let me print them first instead.

```python
both_ids = info[(info.ins_12on>0)&(info.ne_12on>0)].index.tolist()
sel100 = both_ids[:100]
print(len(both_ids))
print(json.dumps([int(s) for s in sel100]))
json.dump([int(s) for s in sel100], open('/work/sel100_both.json','w'))
```