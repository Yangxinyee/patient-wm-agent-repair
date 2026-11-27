`true_in` and `true_ne` take a stay, an hour and a factor, which suggests they return the record-based true shift for one stay-hour. I'll apply `true_in` to the reached insulin rows of the original rollout (fwd_1) for x2 and half. I'll check that the pooled means reproduce my notebook's -13.07 and +6.54, then split the truth by the same rate bands for comparison with the model.

```python
d1=pd.read_csv('/work/fwd_1/dose_edit_original.csv')
out={}
for ed,f in [('insulin_x2',2.0),('insulin_half',0.5)]:
    dd=d1[(d1.edit==ed)&(d1.reached==1)].copy(); dd['tr']=[true_in(s,h,f) for s,h in zip(dd.stay_id,dd.hour)]
    s=dd.groupby('stay_id').tr.mean(); out[ed]=(round(dd.tr.mean(),2), s.groupby(pd.cut(ins.reindex(s.index),[0,1.5,2.5,3.5,20]),observed=True).mean().round(2).tolist())
print(out)
```