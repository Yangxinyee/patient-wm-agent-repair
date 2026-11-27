Run 231 shows that even the counterfactual logits do not change, although the injection fires. So `run_cf` must take z for the state update from somewhere other than the patched `zpost`. Possibly it uses a local reference or the stored tool states. Next I will read `run_cf`'s local names and constants to find where z comes from, so I can inject at the right point. This is a debugging step, so no hypothesis about the model is at stake yet.

```python
c=run_cf.__code__; print(c.co_varnames, [x for x in c.co_consts if not hasattr(x,'co_code')][:12], c.co_names)
c2=step.__code__; print('step G:',[x for x in c2.co_names][:30])
```