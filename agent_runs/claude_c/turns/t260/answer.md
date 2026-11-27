The edit labels are insulin_x2, insulin_half, ne_x2 and ne_half. The error came from `d.shift`, which resolves to the DataFrame method rather than the column. I'll index `d['shift']` and filter on the exact label 'insulin_x2', then rerun the same banded comparison.

```python
def band_tab(f, ed='insulin_x2', rr=ins, cuts=[0,1.5,2.5,3.5,20]):
    d=pd.read_csv(f); d=d[(d.edit==ed)&(d.reached==1)]
    s=d.groupby('stay_id')['shift'].mean(); r=rr.reindex(s.index)
    return s.groupby(pd.cut(r,cuts),observed=True).agg(['mean','count']).round(2).values.tolist(), round(d['shift'].mean(),2)
for k,m in [(1,'original'),(6,'edited'),(9,'edited'),(25,'edited')]:
    print(k, band_tab(f'/work/fwd_{k}/dose_edit_{m}.csv'))
```