The kernel kept no source for these functions, only the functions themselves. I'll read their signatures and call `zcheck2` on one stay to see what it returns. If it reproduces the run-141 error statistics, I can perturb its inputs to locate the z_post mismatch without rewriting the encoder.

```python
for f in ['enc_eu2','zcheck2','gf','fix_atom2','per_bundle_err','split_atoms','enc_bundle2','fbin_f32','zcheck']:
    try: print(f, inspect.signature(globals()[f]))
    except Exception as ex: print(f, ex)
print(globals()[ 'zcheck2'].__code__.co_varnames[:40])
print(globals()['zcheck2'].__code__.co_names[:60])
print(globals()['enc_eu2'].__code__.co_varnames[:30], globals()['enc_eu2'].__code__.co_names[:40])
```