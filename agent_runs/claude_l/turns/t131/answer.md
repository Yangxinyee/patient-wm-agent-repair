The probe structure is clear now. My `z_post` probe made z = P·[ctx, e, u], with 10 rows reading ctx, 11 reading e and 11 reading u. The per-bundle errors are stored in `ee` (for e) and `eu` (for u), and the bundles are in `recs` and `BUN`.

I'll tabulate the e error by bundle composition, using codes and kinds. If the error appears only when a particular code, unit or value range is present, that points to the featurization bug.

```python
print(str(recs[0])[:300])
from collections import defaultdict
grpE=defaultdict(list)
for i,(r) in enumerate(recs):
    atoms = r[2] if isinstance(r,(tuple,list)) and len(r)>2 else None
    try:
        key=tuple(sorted((a['code'],a['kind']) for a in atoms)) if atoms is not None and isinstance(atoms[0],dict) else str(type(atoms))
    except Exception as ex: key='err'
    grpE[key].append(ee[i])
for k,v in sorted(grpE.items(), key=lambda kv:-len(kv[1]))[:14]:
    v=np.array(v); print(k, len(v), 'median %.3g  p10 %.3g  p90 %.3g'%(np.median(v),np.percentile(v,10),np.percentile(v,90)))
```