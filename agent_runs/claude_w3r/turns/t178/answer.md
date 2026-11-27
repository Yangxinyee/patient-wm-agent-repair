The call only failed on dtype: the implementation works in float64. I'll cast all inputs to double and retry the variants. Success is agreement with `step_at` at about 1e-6.

```python
D_ = lambda x: torch.as_tensor(x).double()
for var in [None, 0, 1, 'std']:
    try:
        r_ = core_step(D_(hp), D_(op), dt, D_(EUs[k][0]).reshape(-1), D_(EUs[k][1]).reshape(-1), D_(zk), D_(Mem), D_(Mage), var)
        print(var, [tuple(x.shape) for x in r_] if isinstance(r_, tuple) else r_.shape,
              'err y', float((D_(r_[0])-y_ref.double()).abs().max()), 'err o', float((D_(r_[1])-o_ref.double()).abs().max()))
    except Exception as ex: print(var, 'ERR', repr(ex)[:200])
```