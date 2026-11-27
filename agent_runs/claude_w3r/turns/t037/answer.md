The deviation SD is 0.62, against a rate SD of 1.01. The run then crashed because a few insulin bundles have no prior glucose_finger, so their deviation is NaN. I will drop those rows and rerun the probes of the deviation at bundle lags 0–6. Usable R² at lag 0 that persists over several bundles would mean the state carries the dose deviation.

```python
okd = Bi.dev.notna().values
Bi2 = Bi[okd].reset_index(drop=True)
res_lag={}
for lag in [0,1,2,3,4,6]:
    rows_=[]
    for _,r_ in Bi2.iterrows():
        bb=B[(B.stay_id==r_.stay_id)]; j=np.where(bb.hour.values==r_.hour)[0][0]+lag
        rows_.append(bb.row.values[j] if j<len(bb) else -1)
    rows_=np.array(rows_); ok=rows_>=0; res_lag[lag]=rows_
    r2d=probe(Xs[rows_[ok]], Bi2.dev.values[ok], Bi2.stay_id.values[ok])
    r2c=probe(Xs[rows_[ok]], Bi2.c.values[ok], Bi2.stay_id.values[ok])
    print('bundle lag',lag,'R2 dev %.3f | R2 band-mean c %.3f | n %d'%(r2d,r2c,ok.sum()))
```