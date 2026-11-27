The tuning file has one row per event, with rate and dose columns. For each stay I'll compute the mean recorded insulin rate from hour 12 on. I'll then average each model's insulin-x2 shift over reached stay-hours, within rate bands. A truly linear effect would give a shift proportional to the band's rate; a flattening shift at high rates for N5 would point to saturation.

```python
ins=tu[(tu.event=='insulin')&(tu.hour>=12)].groupby('stay_id').rate.mean()
def band_tab(f):
    d=pd.read_csv(f); d=d[(d.edit.str.contains('insulin'))&(d.edit.str.contains('2'))&(d.reached==1)]
    s=d.groupby('stay_id').shift.mean(); r=ins.reindex(s.index)
    bands=pd.cut(r,[0,1.5,2.5,3.5,20])
    return s.groupby(bands,observed=True).agg(['mean','count']).round(2).values.tolist(), round(s.mean(),2)
print(pd.read_csv(fs[0]).edit.unique())
for k,m in [(1,'original'),(6,'edited'),(9,'edited'),(25,'edited')]:
    print(k, band_tab(f'/work/fwd_{k}/dose_edit_{m}.csv'))
print(pd.cut(ins.reindex(pd.read_csv(fs[0]).stay_id.unique()),[0,1.5,2.5,3.5,20]).groupby(pd.cut(ins.reindex(pd.read_csv(fs[0]).stay_id.unique()),[0,1.5,2.5,3.5,20]),observed=True).size().tolist())
```