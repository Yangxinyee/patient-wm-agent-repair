Numeric ×5 barely changes the model's lag-1 sensitivity to within-tier dose deviations: -0.14 per U/h, against roughly -2 implied by the records. Yet it costs 0.0016 nats. So its gain in the dose-edit readout comes from the response to out-of-range doses, not from learning the causal effect. Its cost per unit of readout gain is still about 3× better than head-0 scaling.

I'll test the modest combination: numeric ×5 plus head-0 ×2.5 (48 basis 'in' ops, alpha 1.5, on prelude attn.out columns 0–47). The expected main readout is about -2.9 to -3.0 if the two gains compound. A result near -2.5 would mean they don't stack, and head scaling should be dropped.

```python
for j_ in range(48):
    e_ = np.zeros(192); e_[j_]=1.0; rank1('core.prelude.0.attn.out.weight', e_, 1.5, side='in')
print(len(ops()), ops()[0]['name'], ops()[2]['name'], ops()[-1]['alpha'])
```