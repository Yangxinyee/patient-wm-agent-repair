The kernel already has `make_bundles`, `run_tf`, `mystep`, `free_run`, `scale_atoms`, `head_mean`, `probs_from_v0` and `true_shift_rows`. `free_run` may be a local rollout from earlier. I'll look at the signatures and the names each function uses so I can rebuild a local check against the tool's hour-13 predictions.

```python
import inspect
for f in ['make_bundles','run_tf','mystep','free_run','scale_atoms','ins_atoms_scaled','head_mean','head_mean_rows','head_probs','probs_from_v0','v0_of','view0','build_edit','target_readout','true_shift_rows','n3_of','stay_views','head_inputs','fine_bin','setW','resetW','nll_of']:
    fn = globals()[f]
    try: sig = str(inspect.signature(fn))
    except Exception as ex: sig = '?'
    co = fn.__code__
    print(f, sig, '| names:', co.co_names[:40], '| consts:', [c for c in co.co_consts if isinstance(c,(str,int,float)) and not isinstance(c,bool)][:25])
    print('---')
```