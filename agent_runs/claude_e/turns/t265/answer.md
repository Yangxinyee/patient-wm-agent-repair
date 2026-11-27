On tu2 (2020000300..399) the −2.6 edit gives insulin x2 −10.80 and x1/2 +5.09, with NE unchanged. I now have −2.6 rollouts on all four tuning batches: f3 from fwd_12, tu1, tu0 and tu2.

I will compute the record-based truth for every batch and the pooled edited-to-truth ratio, weighted by stay-hours, and log the results. A pooled insulin ratio near 0.9 would justify a modestly larger gain, if the factual risk stays acceptable. A ratio near 0.95 or above means −2.6 is final.

```python
allE=[]
for k,lab in [(12,'f3'),(22,'tu1'),(23,'tu0'),(24,'tu2')]:
    d_=pd.read_csv(f'/work/fwd_{k}/dose_edit_edited.csv'); d_['batch']=lab
    dor=pd.read_csv(f'/work/fwd_{k}/dose_edit_original.csv')
    d_=d_.merge(dor[['stay_id','hour','edit','shift']].rename(columns={'shift':'shift_o'}),on=['stay_id','hour','edit'])
    d_=d_[d_.reached].copy()
    fnm={'insulin_x2':(truth_ins,2.0),'insulin_half':(truth_ins,.5),'ne_x2':(truth_ne,2.0),'ne_half':(truth_ne,.5)}
    d_['truth']=[fnm[e][0](s,h,fnm[e][1]) for s,h,e in zip(d_.stay_id,d_.hour,d_.edit)]
    allE.append(d_)
AE=pd.concat(allE,ignore_index=True)
tb=AE.groupby(['batch','edit']).agg(n=('shift','size'),edited=('shift','mean'),orig=('shift_o','mean'),truth=('truth','mean')).round(3)
tb['ratio']=(tb.edited/tb.truth).round(3); print(tb.to_string())
pool=AE.groupby('edit').agg(n=('shift','size'),edited=('shift','mean'),orig=('shift_o','mean'),truth=('truth','mean'))
pool['ratio']=pool.edited/pool.truth; print(pool.round(3).to_string())
with open('/work/notes.md','a') as f:
    f.write("- dose_edit at -2.6 on all 4 tuning batches (fwd_12 f3, fwd_22 tu1, fwd_23 tu0, fwd_24 tu2). Pooled (edited/orig/truth):\n"+pool.round(3).to_string()+"\n")
```