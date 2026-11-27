The loop ran fast (2406 drug bundles in 3.3 s). Building the rate array failed because some `rate` fields aren't plain floats. I'll look at the types and examples of the rate field in the start atoms, then coerce them to floats and rerun the probe. This is a data-format fix, not yet a test of the hypothesis.

```python
from collections import Counter
print(Counter(type(r['ri']).__name__ for r in recE), Counter(type(r['rn']).__name__ for r in recE))
ex_ = [r['ri'] for r in recE if not isinstance(r['ri'], (float,int))][:3]; print('examples', ex_)
at_ = [a for a in B0[ks[0]][1]] if False else [a for r in recE[:1] for a in stay_data(r['stay'])[0][r['k']][1]]
print(at_)
```