The run failed because my earlier `ln` helper uses the global name `F` for `torch.nn.functional`. My regression loop in run 72 overwrote `F` with a numpy array. I'll restore `F` and rerun the same variant search. The outcome I'm looking for is the same: one configuration should match the GPU state to about 1e-5.

```python
F = Fn
r_ = bundle_vecs(bl[0][1]); print(type(r_), [getattr(t,'shape',None) for t in (r_ if isinstance(r_,(tuple,list)) else [r_])])
for mode in ['post','pre']:
    for au in [1.0, 60.0, 3600.0]:
        for mc in [True, False]:
            try: print(mode, au, mc, run_variant(mode, au, mc))
            except Exception as ex_: print(mode, au, mc, 'ERR', repr(ex_)[:200])
```