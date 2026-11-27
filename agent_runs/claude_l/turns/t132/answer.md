`recs` holds per-bundle tuples: stay, bundle index, two small counts, then the probe's true e and u projections. The atoms themselves are in `BUN[stay][k]`.

The overall e error is large: median 1.57, p10 0.53, p90 2.31. I'll redo the grouping by atom composition using `BUN`, to see which kinds of bundle carry the error.

```python
grpE=defaultdict(list)
for i,r in enumerate(recs):
    s,k=r[0],r[1]
    try:
        atoms=BUN[s][k][1]
        key=tuple(sorted((a['code'],a['kind']) for a in atoms))
    except Exception as ex:
        key=('err',str(ex)[:30])
    grpE[key].append(ee[i])
for k_,v in sorted(grpE.items(), key=lambda kv:-len(kv[1]))[:16]:
    v=np.array(v); print(k_, len(v), 'median %.3g  p10 %.3g  p90 %.3g'%(np.median(v),np.percentile(v,10),np.percentile(v,90)))
```