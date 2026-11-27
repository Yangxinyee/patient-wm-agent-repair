Run 275 failed inside the kernel's own `datom`, not in my new code. In the last few runs I redefined the global `T` as a small converter function. The earlier helper functions evidently look tensors up in a global dict called `T`, so I overwrote it.

I will restore `T` as a dict of the original tensors and rename my converter `TT`. Then I will repeat the check that my encoder reimplementation matches `u_rec`, and print the per-code normalisation and bin edges for the two drugs from `VOCAB['numeric']`, which is a list.

If the match is at rounding level, I proceed to build the per-drug amplification.

```python
import numpy as np
TT = lambda x: x.numpy() if hasattr(x, 'numpy') else np.asarray(x)
T = {n: TT(load_original(n)).astype(np.float64) for n in NAMES}
for e in VOCAB['numeric']:
    nm = e.get('name', e.get('code', '')) if isinstance(e, dict) else e
    if isinstance(e, dict) and any(str(v) in ('insulin', 'norepinephrine', '8', '9') for v in [e.get('name'), e.get('code'), e.get('id')]):
        print(str(e)[:900])
print('first entry keys', list(VOCAB['numeric'][0].keys()) if isinstance(VOCAB['numeric'][0], dict) else VOCAB['numeric'][0])
for args in [(2.0, 0.1), (0.0, 0.0), (6.0, 0.3)]:
    ur = np.asarray(u_rec(*args), dtype=np.float64).ravel()
    V = np.stack([np.asarray(datom('insulin', args[0]), dtype=np.float64).ravel(), np.asarray(datom('norepinephrine', args[1]), dtype=np.float64).ravel()])
    u_my = enc_int(V)[3]
    print(args, 'shapes', ur.shape, V.shape, 'my-vs-u_rec maxdiff %.2e' % np.abs(u_my - ur).max())
```