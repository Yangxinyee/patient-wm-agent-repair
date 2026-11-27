The last run gave the full tensor layout:
- The value head is a two-layer MLP on `with_code`, plus a per-code 17×17 transition table.
- `z_post` maps [ctx, e_k, u_k] (576 inputs) to a 32-dim mean and log-sd.
- The recurrent cell is a linear state-space-style update (`b_in`, `a_in`, `delta_in`) followed by `c_out` and a one-layer coda.

So dose information has two routes into the state: the u_k token in the prelude, and the z posterior.

One hypothesis worth testing first: the z posterior at action bundles may already encode the dose's departure from what the model's prior expects. If so, it is a ready-made "dose deviation" signal, and amplifying it could add the missing effect without double-counting in factual predictions.

To test this I need to recover what my helper functions and the fwd_3 states contain, because that context was lost in the restart. This run only inspects, it does not edit anything.

```python
import inspect, json
for fn in ['band','true_shift_rows','stay_atoms','atom_vec2','encode','ctx_v','zpred','deconv']:
    try:
        print('-----', fn); print(inspect.getsource(globals()[fn])[:1500])
    except Exception as ex: print(fn, ex)
print(S.files, json.load(open('/work/fwd_3/summary.json')) if True else None)
print(len(selT), len(sel1), np.unique(nat, return_counts=True), mask.sum(), Kins.round(2), Knor.round(2))
```