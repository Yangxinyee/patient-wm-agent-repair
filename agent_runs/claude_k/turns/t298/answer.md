`u_rec` now runs again, so restoring `T` fixed the shared dependency. `datom` rejects the string 'insulin', so it is keyed some other way, probably by code id.

The vocab gives the realistic ranges:
- insulin rate: median 1.5 units/h, bin edges 0.5 to 5;
- norepinephrine rate: median 0.46, bin edges 0.03 to 1.0 mcg/kg/min;
- norepinephrine dose: evidently the rate converted to mg per hour, median 2.2.

I will read `datom`'s and `u_rec`'s code constants to see how they key the code. Then I will redo the encoder match test. If the match is exact, I can build the amplification.

```python
import numpy as np
print('datom consts', datom.__code__.co_consts[:40]); print('datom names', datom.__code__.co_names)
print('u_rec consts', u_rec.__code__.co_consts); print('u_rec names', u_rec.__code__.co_names)
cands = [8, 9, 'ins', 'ne', 'I', 'N']
for c in cands:
    try:
        a = np.asarray(datom(c, 2.0), dtype=np.float64); print('datom', repr(c), 'ok', a.shape)
    except Exception as ex: print('datom', repr(c), 'fail', type(ex).__name__, ex)
```