Key finding: u holds both rates almost perfectly linearly (insulin R² 0.995, norepinephrine R² 0.984), but E doesn't. The rate information is clean in u and gets mixed with the huge h only when it enters the slot tokens. So the dose channel should be built from u, inside the prelude, and routed to the head.

Before engineering it, I want a cheap way to price candidate head-level corrections δ (the added prediction shift) on factual fit. I'll use the original model's per-reading predicted means and residuals: ΔNLL ≈ (δ² − 2δ·e)/(2σ²). Next I'll check what the factual CSVs in fwd_1/fwd_3 contain (columns, number of rows, residual sd by event).

```python
import os, glob
for d in ['/work/fwd_1','/work/fwd_2','/work/fwd_3']:
    print(d, os.listdir(d))
F3 = pd.read_csv(glob.glob('/work/fwd_3/factual_*.csv')[0]); print(F3.columns.tolist(), F3.shape)
print(F3.head(3))
if 'pred_mean' in F3.columns and 'value' in F3.columns:
    F3['res'] = F3['value'] - F3['pred_mean']
    print(F3.groupby('event')['res'].agg(['count','mean','std']))
```