The keys (`sc`, `mnorm`, `kf`, `pre`/`post`) are an older set of implementation variants. `step_at` instead calls `core_step_v(hprev, dt, e, u, zk, Mem, Mage, dec_mode='std')`, the validated version that takes no `outprev`. I'll call `core_step_v` with double inputs and compare it with `step_at`. A match gives me the step function for a recursive runner.

```python
try:
    r_ = core_step_v(D_(hp), dt, D_(EUs[k][0]).reshape(-1), D_(EUs[k][1]).reshape(-1), D_(zk), D_(Mem), D_(Mage))
    print(type(r_), [tuple(x.shape) for x in r_] if isinstance(r_, tuple) else r_.shape)
    for i, x in enumerate(r_ if isinstance(r_, tuple) else [r_]):
        x = D_(x)
        for nm, ref in [('y', y_ref), ('o', o_ref)]:
            if tuple(x.shape)==tuple(ref.shape): print(i, nm, float((x-ref.double()).abs().max()))
except Exception as ex: print('ERR', repr(ex)[:300])
```