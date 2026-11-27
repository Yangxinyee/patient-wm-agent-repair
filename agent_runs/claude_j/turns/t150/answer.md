The replica matches the tool for the first bundles (error 4e-5) but not at bundles 5 and 8 (0.9, 0.35). `one_step_err` probably builds the memory differently by default, so the memory matters by then. `run_variant3` handles memory explicitly through its `memfill` argument and the skip rule. I'll run it over a full stay with a few candidate `memfill` values and keep whichever gives tiny errors throughout. That variant then becomes the base of my counterfactual runner.

```python
for mf in [True, False, 'e', 'zero', 'skip']:
    try:
        r_=run_variant3(mf, kmax=len(BUN[st_]), stay=st_); r_=np.asarray(r_)
        print(repr(mf), 'n', r_.size, 'max err', float(np.max(r_)) if r_.size else None, 'median', float(np.median(r_)) if r_.size else None)
    except Exception as ex_: print(repr(mf), 'err', type(ex_).__name__, str(ex_)[:80])
```