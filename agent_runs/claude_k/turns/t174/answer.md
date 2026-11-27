The README confirms that every infusion record has its own `dose` column:
- Insulin dose is in units and equals rate × 1 h.
- Norepinephrine dose is in mg, rate × weight × 60/1000. That gives ×4.8 at 80 kg, matching the vocab medians.

If weight varies by stay, norepinephrine dose is not a fixed multiple of rate. That alone could explain part of the norepinephrine mismatch.

My filter crashed on NaN values. I'll redo it by matching the event name. I'll print the drug rows of stay 1010001440 around the bundle, and check dose/rate ratios across the loaded rows.

```python
dr=rows_[rows_.event.isin(['insulin','norepinephrine'])]
print('t_j0',t_j0); print(dr[(dr.hour>t_j0-2)&(dr.hour<t_j0+2)].to_string()[:1500])
d_ins=it[it.event=='insulin']; d_ne=it[it.event=='norepinephrine']
print('insulin dose/rate', (d_ins.dose/d_ins.rate).describe().round(4).to_dict())
print('ne dose/rate', (d_ne.dose/d_ne.rate).describe().round(4).to_dict())
print('ne dose/rate by stay sd (first 20 stays):', d_ne.assign(q=d_ne.dose/d_ne.rate).groupby('stay_id').q.agg(['mean','std']).head(8).round(4).to_dict())
```