The kernel holds a full dataset-based teacher-forced pipeline: `build_ds(df, stays, seed)`, `tf_states_ds`, `tf_eval(P, ds, ret_states)`, `tf_nll`, `kill_dims` and `fit_eval`. `tf_states` shows the loop structure:
- The batch runs over stays, bundle by bundle.
- Each step computes ctx, then the z posterior mean, then memory of past e's with ages in seconds, then `step_b`.

So I can write an instrumented copy of the loop that calls `step_b(..., ret=True)` and collects the prelude attention weights and value inputs at each bundle. That is what I need to fit the decoder that stages the insulin rate.

To write that copy I need the exact layout of a `build_ds` dataset (keys and shapes) and the constants each pipeline function reads. I'll build a tiny dataset on 3 training stays and list its contents, plus the string and numeric constants and global names used by `tf_states_ds`, `tf_eval`, `fit_eval` and `kill_dims`.

```python
ds_ = build_ds(tr, sel2[:3], 0)
print(type(ds_).__name__, (list(ds_.keys()) if isinstance(ds_, dict) else len(ds_)))
if isinstance(ds_, dict):
    for k_,v_ in ds_.items():
        print(' ', k_, type(v_).__name__, getattr(v_,'shape', (len(v_) if hasattr(v_,'__len__') else v_)))
else:
    for i_,v_ in enumerate(ds_): print(' ', i_, type(v_).__name__, getattr(v_,'shape', (len(v_) if hasattr(v_,'__len__') else v_)))
for f_ in [tf_states_ds, tf_eval, fit_eval, kill_dims, tf_nll]:
    co=f_.__code__
    print(f_.__name__, co.co_varnames[:co.co_argcount], '| consts', [c for c in co.co_consts if isinstance(c,(str,int,float))][:40], '| names', co.co_names[:45])
```