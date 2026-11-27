Both edited dicts produce identical states, and both differ from the original by up to 55 in h. Yet `P_` leaves all core blocks untouched. So the core step is not reading everything from the dict it is passed. Some inner function probably uses a fixed global weight set (likely the scaffold) for part of the computation. That would make every offline core evaluation since then unreliable.

I'll inspect which globals `step_b`, `ctx_fn` and `zpost_mean` reference. A global dict name such as `PS`, `Wt` or `TJ`, or precomputed arrays, would explain it; then I can rebind or wrap them so the passed dict is used throughout.

```python
for f in [step_b, ctx_fn, zpost_mean, core_step_b, head_value, mha_b, ffnb, LNb]:
    c=f.__code__
    print(f.__name__, 'args', c.co_varnames[:c.co_argcount], '| globals', [n for n in c.co_names if n not in ('torch','np','len','range','float','int','zip','enumerate','max','min','sum','list','tuple','dict')][:30])
    if f.__defaults__: print('   defaults', [ (type(d).__name__, (len(d) if hasattr(d,'__len__') else d)) for d in f.__defaults__])
```