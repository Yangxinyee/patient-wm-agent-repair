Stops are not a usable source of variation. They are mechanical: a stop happens only once the rate has titrated down to ≤ 0.05 mcg/kg/min (always), or 0.05–0.1 (22%). Above that, the rate is never stopped. The rate-based titration innovation remains the only clean variation, and the norepinephrine x2 estimate stays at about −0.3 ± 0.15 mmHg. I judge "absent" (true readout 0) somewhat more likely than a tiny negative effect, since a designed "small" effect would probably be clearly detectable.

My leading plan is therefore to make the model exactly insensitive to drug dose values. Then the edited and factual rollouts coincide and all four readouts are exactly 0, while factual fit is kept by preserving each drug's average embedding.

To do that I need to know how drug atoms are embedded: which categorical rows (unit, rate_unit) belong to which drug, and whether END atoms carry rate and dose. Next I will read tensors.json for any descriptions, and compute from the weights how large the dose-dependent embedding terms are relative to the rest.

```python
TJ = json.load(open('/model/tensors.json'))
ks = list(TJ.keys())[:5]
for k in ks: print(k, json.dumps(TJ[k])[:300])
for k in ['embed.numeric.weight','embed.bins.dose.weight','embed.bins.rate.weight','embed.categorical.rate_unit.weight','embed.categorical.unit.weight','embed.kind.weight','embed.categorical.action.weight','embed.categorical.status.weight','embed.kq.weight','embed.text.weight']:
    print(k, json.dumps(TJ.get(k))[:400])
s = open('/model/tensors.json').read(); print(len(s)); print(s[:1500])
```