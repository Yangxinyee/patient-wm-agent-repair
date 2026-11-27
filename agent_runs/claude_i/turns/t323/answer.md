`run_stay` replays one stay's bundles from encoded e/u arrays, using the weights in `WT`. That means I can test weight edits locally on stays from forward pass 3, without spending forward passes.

**Next:** I need the scaled-dose case too. I need helpers that rebuild a drug bundle's u with a changed rate, plus the per-bundle info (stay, hour, rates). This step prints only their argument lists and the keys of one `INFO` entry.

If a helper such as `u_for` or `u_only_x` takes a rate factor, I can build the full local test harness in the next run.

```python
import types
for k in ['u_for','u_only','u_only_x','step_u','bundle_eu','eu_routed2','eu_with_end','atom_vec','build_bundles','encode_bundles','drug_after','tgt_dev','dev_ins','true_shift','panel','one_step','preds_next','next_pred','run_stay_z','head_value2','pred_mean_v','view_of']:
    f = globals().get(k)
    if isinstance(f, types.FunctionType):
        c = f.__code__; print(k, c.co_varnames[:c.co_argcount], [n for n in c.co_names if n.isupper() or n in ('INFO','RI','RN','dts','tmin')][:8])
print('INFO[0]', {k:(type(v).__name__) for k,v in (INFO[0].items() if isinstance(INFO[0],dict) else [])} if isinstance(INFO[0],dict) else type(INFO[0]).__name__, str(INFO[0])[:300])
print('dts', type(dts).__name__, getattr(dts,'shape',None), 'tmin', type(tmin).__name__, getattr(tmin,'shape',None), 'Ee', type(Ee).__name__, tuple(Ee.shape))
print('RI item', str(next(iter(RI.items())))[:200]); print('RN item', str(next(iter(RN.items())))[:200])
```