The vocab and tensor list point to a likely bug: several embedding tables are missing from my replica's atom vector:
- `categorical.action`, `discharged_to`, `route`, `status`, `text`;
- possibly the wrong `kq` row.

The note also fixes z = (x − median)/iqr clipped to ±10, and the bin rule. If those tables' row 0 is nonzero, or measurements use a different `kq` or `status` row, that would explain a constant error on measurement bundles while action atoms still match.

I'll print the row norms of these small tables and list the functions I defined in the kernel, so I can find my probe-comparison routine and test fixes.

```python
for n in ['embed.kq.weight','embed.categorical.action.weight','embed.categorical.discharged_to.weight','embed.categorical.route.weight',
          'embed.categorical.status.weight','embed.text.weight','embed.categorical.unit.weight','embed.categorical.rate_unit.weight','embed.kind.weight','embed.ekind.weight']:
    w=np.asarray(load_original(n)); print(n, np.round(np.linalg.norm(w,axis=1),3))
import types
fns=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and getattr(v,'__module__','')=='__main__']
print(fns)
```