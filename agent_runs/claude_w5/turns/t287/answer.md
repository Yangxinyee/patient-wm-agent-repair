Big progress: with the kv tensor saved, the edit now works for the dose increases.

| Readout | Edited | Original | Target (g-est) |
|---|---|---|---|
| insulin x2 | **−9.43** | −0.55 | −14.3 |
| insulin x1/2 | **−1.63** (wrong sign) | 0.00 | +7.15 |
| norepinephrine x2 | **+3.21** | +0.19 | +4.11 |
| norepinephrine x1/2 | **−0.40** (much too weak) | −0.17 | −2.05 |

For a response linear in the dose change, x1/2 should be about −½ of x2. Splitting the observed numbers into an odd part (linear) and an even part (responding to |change|):
- insulin: linear part about −3.1 per unit, even part about −6.4 per unit;
- norepinephrine: +2.0 and +1.2.

In effect the edit responds strongly to dose increases and hardly at all to decreases. The per-bundle test in run 260 showed the same pattern. Likely sources are a convex nonlinearity in the chain: the rate code in u, the stage-1 read, or the GELU gate in stage 2.

Next I'll locate it locally on training-stay insulin bundles. I'll check whether `mystep` exposes intermediate values through its `R` argument, then trace the injection across dose factors 0.5–2.

```python
import numpy as np
j = 3; d = S_[j]
old = WCUR; WCUR = WE
try:
    R = {}
    r_ = mystep(d, R=R)
    print('R type', type(R).__name__, 'keys', list(R.keys())[:30] if isinstance(R, dict) else None)
    for k_, v_ in (R.items() if isinstance(R, dict) else []):
        print('  ', k_, getattr(np.asarray(v_), 'shape', None))
except Exception as ex:
    import traceback; traceback.print_exc()
finally:
    WCUR = old
import pickle
EP_ = pickle.load(open('/work/edit_parts.pkl','rb'))
print('edit_parts keys', list(EP_.keys()) if isinstance(EP_, dict) else type(EP_))
for k_, v_ in (EP_.items() if isinstance(EP_, dict) else []):
    print('  ', k_, type(v_).__name__, getattr(v_, 'shape', v_ if np.isscalar(v_) else None))
```