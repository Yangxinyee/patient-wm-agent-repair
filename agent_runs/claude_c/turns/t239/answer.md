The run found no rows only because the edit labels are `ne_x2` and `ne_half`, not the names I used. I will rerun the same NE rate-band split with the right labels.

```python
for ed,f in [('ne_x2',2.0),('ne_half',0.5)]:
    r=o1[(o1.edit==ed)&o1.reached].copy()
    mx=r.stay_id.map(lambda s: neg[s].rate.max() if s in neg else 0)
    r['grp']=pd.cut(mx,[-1,0.5,0.999,1.01],labels=['<=.5','.5-1','cap'])
    r['tru']=[true_ne(s,h,f) for s,h in zip(r.stay_id,r.hour)]
    print(ed, len(r), r.groupby('grp',observed=True).agg(n=('shift','size'),model=('shift','mean'),tru=('tru','mean')).round(2).to_dict('index'))
```