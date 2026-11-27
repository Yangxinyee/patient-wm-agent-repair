The per-hour breakdown points to a different problem than I expected. The edited model's *factual* rollout drifts: glucose is +3.0 above the original at hour 13 and +13.5 by hour 24, and MAP is −1.5 at hour 13 and −7.1 by hour 24. The wrong-signed dose shift also grows over time.

The hour-13 prediction comes from the record through 12:05, which is identical to teacher forcing. So the +3.0 offset at hour 13 should be close to zero if my local implementation matches the tool. I have three explanations:
- (a) the tool computes my edit differently, for example in reduced precision on the GPU, where the 25·s gate offset no longer cancels between the twin units;
- (b) the decoded residual x is strongly biased on these tuning stays;
- (c) a sign or convention mismatch.

To check, I'll reproduce the tool's hour-13 predictions locally for the tuning stays. First I need to find the helper functions in my kernel, including the bundle builder.

```python
import types, inspect
fns = sorted(k for k,v in globals().items() if isinstance(v, types.FunctionType) and not k.startswith('_'))
print(fns)
others = sorted((k, type(v).__name__) for k,v in globals().items() if not k.startswith('_') and not isinstance(v, (types.FunctionType, types.ModuleType, type)))
print(others)
for f in fns:
    try:
        print(f, inspect.signature(globals()[f]))
    except Exception as ex:
        print(f, 'sig?', ex)
```