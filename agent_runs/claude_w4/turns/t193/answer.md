The replica of the recurrent step is only approximate (relative error 13–15%, R2 about 0.99). It is good enough for rough linear sensitivities but not for exact design. So I'll inject inside the prelude, whose CPU replica is exact, and calibrate end-to-end with forward passes.

First I need the scale of the prelude's pre-norm residual stream at drug bundles. The injection passes through `prelude_norm`, so its effect on E scales as 1/std(x). If std(x) is fairly constant across bundles and slots, a fixed injected vector gives a predictable change in E. I'll check what `prelude2` returns on the stored drug-bundle tokens TOK.

```python
print('TOK', type(TOK).__name__, tuple(TOK.shape) if hasattr(TOK,'shape') else len(TOK))
W_ = {n: load(n) for n in NAMES if n.startswith('core.prelude') or n.startswith('core.slots')}
r2 = prelude2(TOK[:5] if hasattr(TOK,'shape') else TOK, W_)
print(type(r2).__name__, [tuple(x.shape) if hasattr(x,'shape') else type(x).__name__ for x in (r2 if isinstance(r2,(tuple,list)) else [r2])])
print([n for n in NAMES if n.startswith('core.prelude')])
```