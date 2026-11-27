The older helper reaches 0.10 using `atom_vec` and `set_enc_`, so my newer `enc_bundle` (with options, from the run-152 experiments) is probably the worse encoder. Time units match: minutes × 60 → seconds, log(dt)/10. Next I will look at the signatures and internals of `atom_vec` and `set_enc_`. Then I can rebuild u_k with the better encoder and see whether drug bundles reach about 0.1 error.

```python
for fn in ['atom_vec','set_enc_','mlp_','ln_']:
    f=globals().get(fn)
    if f is None: print(fn,'missing'); continue
    c=f.__code__; print(fn, inspect.signature(f), c.co_names[:40], [k for k in c.co_consts if not hasattr(k,'co_code')][:25])
```