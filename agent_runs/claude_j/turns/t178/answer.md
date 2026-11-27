`BUN[stay]` is a list of (minute, [(event, kind, row), ...]) bundles, built straight from the record.

To run the replica forward under changed doses I need three things: what `step` returns, how the initial state is set, and how z is computed. Helpers like `ctx_of`, `z_err_single`, `run_variant3` probably already do parts of this. I will print the signatures of those helpers and run `step` once on the first bundle to see its return structure.

```python
for k in ['ctx_of','z_err_single','z_err_v','z_err4','enc_e','step0v','run_variant3','my_one2','zs_','zsc','one_step_err','norm_bun']:
    c=globals()[k].__code__; print(k, c.co_varnames[:c.co_argcount], [n for n in c.co_names if n in globals()][:8])
print([n for n in NAMES if any(t in n for t in ['gap','type_emb','query_norm','filter','pending','time_head'])][:20])
```