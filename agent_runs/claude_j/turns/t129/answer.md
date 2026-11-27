The kernel still has the replica (`build_bundles`, `step`, `run_variant*`), the head functions (`view_parts`, `head_mean`, `q_for`), and the newest helpers from the lost replies (`reading_table`, `add_targets`, `cexp`, `hourly_maps`, `cross_val_predict`). So the last session was building a table of readings with their head query vectors q and regression targets. I'll check signatures and the key dictionaries so I can reuse them without re-deriving. Printing only signatures and key lists keeps the output short.

```python
import inspect
for f in ['build_bundles','step','run_variant3','one_step_err','view_parts','head_mean','zsamp_view','head_mean_avg','ext_for','norm_bun','build_reading_table','reading_table','q_for','cexp','hourly_maps','add_targets','cross_val_predict']:
    g_=globals()[f]
    try: sig=str(inspect.signature(g_))
    except Exception as ex_: sig='?'
    print(f, sig[:110], '| vars:', ','.join(g_.__code__.co_varnames[:14])[:120])
for dn in ['HI','HN','GFH','CFG','cfg','BUN','ROWS','W']:
    dv=globals()[dn]; ks=list(dv.keys())
    print(dn, len(ks), [str(x)[:25] for x in ks[:6]])
```