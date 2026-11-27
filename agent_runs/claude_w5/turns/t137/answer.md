The vocab confirms the predicted mean is Σ_{b=1..16} p(b)·centre[b−1], with z-scores robust per code. The 15 head types and the 8-input register layout are still undocumented, so I will find them by matching the tool's predicted means.

First I need to know what my kernel already holds:
- which data frames and arrays exist;
- what `make_bundles` returns;
- the local variable names of `tf_check`, `head_inputs` and `head_probs`. Their source cannot be printed, but their code objects show how they are wired.

This lets me write a clean per-stay teacher-forced loop plus head instead of guessing the interfaces of old functions.

```python
import types, numpy as np
big = {k: type(v).__name__ + (str(getattr(v,'shape','')) if hasattr(v,'shape') else (f'[{len(v)}]' if hasattr(v,'__len__') else ''))
       for k,v in globals().items() if not k.startswith('_') and not callable(v) and not isinstance(v, types.ModuleType)}
print({k:v for k,v in big.items() if any(t in v for t in ['DataFrame','ndarray','dict','list','Tensor'])})
for fn in ['tf_check','head_inputs','head_probs','make_bundles','cstep','enc2']:
    co = globals()[fn].__code__
    print(fn, 'vars:', co.co_varnames[:30], 'names:', [n for n in co.co_names][:40])
```