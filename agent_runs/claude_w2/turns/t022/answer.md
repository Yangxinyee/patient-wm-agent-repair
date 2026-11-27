Both effects look roughly linear in dose, though NE is less clean:
- **NE:** the 1-hour slope is 3.5-4.0 mmHg per mcg/kg/min from 0.2 to 1.0, slightly lower (2.6-2.9) below 0.2. It is similar across MAP levels (3.3-4.2), so a linear model is a reasonable first approximation. The low-rate dip may be noise or mild nonlinearity.
- **Insulin:** within-band sub-range slopes scatter around −2 (−1.1 to −3.3, noisy because each sub-range has little dose variation), with no clear trend.

Now I need the model's current answers. I'll pick 100 tuning stays that have insulin and/or NE from hour 12 on, favouring stays with both so one request gives all four readouts. I'll run dose_edit on the original model, then compare its predicted shifts with my record-based estimate of the truth on the same stays. This first step only checks the composition of the tuning set.

```python
gt2 = gt.copy()
ti = gt2[(gt2.h>=12)].groupby('stay_id').agg(ins=('ins', lambda s: s.notna().sum()), nor=('nor', lambda s: s.notna().sum()))
print(len(ti), (ti.ins>0).sum(), (ti.nor>0).sum(), ((ti.ins>0)&(ti.nor>0)).sum(), ((ti.ins==0)&(ti.nor==0)).sum())
both_ids = ti[(ti.ins>0)&(ti.nor>0)].index.tolist()
ins_only = ti[(ti.ins>0)&(ti.nor==0)].index.tolist()
nor_only = ti[(ti.ins==0)&(ti.nor>0)].index.tolist()
none_ids = ti[(ti.ins==0)&(ti.nor==0)].index.tolist()
print(len(both_ids), len(ins_only), len(nor_only), len(none_ids))
```