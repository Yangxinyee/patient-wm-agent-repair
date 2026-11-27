sim.py shows I already have a local teacher-forced stay runner. It uses `ctx_from`, `zpost_mean` and `core_step`, where `core_step` also attends over a memory of the last 128 measurement encodings. The dtype error came from mixing the float64 weight dict `WT` with float32 inputs, so I will cast inputs before calling it.

I also have saved token-space Jacobians (Jtok: 128 bundles × 192 token dims × 4 outputs, with baseline predictions base4). These are exactly what I need to pick an injection direction, but my notebook never recorded what the 4 outputs are or which token they refer to.

The function source can't be retrieved, but signatures and docstrings might be. Next I'll print the signature and docstring of the key helpers (Jacobian, prediction, drug-step and policy functions) so I can reuse them rather than rebuild them with my few remaining runs.

```python
import inspect
L=['core_step','jac','jac_at','best_dir','preds_after','preds_next','next_pred','step_ku','step_u','step_k','u_for','dev_ins','tgt_dev','true_shift','eff','run_stay_z','build_bundles','encode_bundles','head_pred','pred_mean_s','pibar','view_ctx','one_step','u_from_V','drug_after','scan','check','rep']
for n in L:
    f=globals().get(n)
    if f is None: print(n,'MISSING'); continue
    try: sig=str(inspect.signature(f))
    except Exception as ex: sig='?'
    d=(f.__doc__ or '').strip().split('\n')[0][:110]
    print(n+sig[:90], '|', d)
print(type(WT['z_in.weight']), WT['z_in.weight'].dtype, type(dts), dts.dtype)
```