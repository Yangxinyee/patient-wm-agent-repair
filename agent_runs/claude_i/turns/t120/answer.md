The regression from before the restart answers the key question. The factual residual (observed minus the model's predicted glucose) still falls with recent dose deviations:
- **1 h after the dose:** -1.59 ± 0.49 per u/h.
- **2 h after the dose:** -1.74 ± 0.45 per u/h.

The true direct effects are about -2.0 and -2.4. So the original model uses only about a quarter of the within-range dose information. That has two consequences:
1. Its weak response to doubled insulin is not just bad extrapolation. It barely responds to dose size at all.
2. An edit that gives it the true, linear dose effect should not hurt the factual fit. If calibrated well, it should slightly improve it.

Next I need the exact structure of the core and heads (tensor names and shapes, config), to decide where to build a dose channel that is linear in dose and has the right time profile. My plan is a torch re-implementation of the full step, checked against the saved states, so I can test edits offline. Keeping output compact.

```python
cfgv = VOCAB.get('config', {})
print({k: (v if not isinstance(v,(list,dict)) else type(v).__name__) for k,v in cfgv.items()})
grp = {}
for nm_ in NAMES:
    top = '.'.join(nm_.split('.')[:2])
    grp.setdefault(top, []).append(nm_)
print(len(NAMES), {k: len(v) for k,v in grp.items()})
print(' '.join(f"{n_[5:]}:{'x'.join(map(str,SHAPES[n_]))}" for n_ in NAMES if n_.startswith('core.')))
```