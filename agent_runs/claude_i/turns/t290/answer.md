The z_post reproduction is exact (median error 1e-7). There are 50 dead hidden units, with pre-activations averaging −5.9, so I can reuse them without changing factual behaviour. z dimensions 18 and 28 vary very little (sd ≈ 0.05).

`step_local` failed on an internal lookup, so I'll try `run_stay_z(ks, E_, U_, zadd)` instead, which appears to run a whole stay with additions to z. Next I'll:
- check whether it reproduces the stored states;
- see which existing arrays (`Fi`, `Fn`) hold per-bundle drug rates;
- look at the record columns, to build exact targets.

```python
print('bundles', len(sid3), 'ins', int(hI.sum()), 'ne', int(hN.sum()), 'stays', len(np.unique(sid3)), 'hours', hr3.min(), hr3.max())
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print('Fi ins rows', N(Fi)[hI==1][:2], 'Fn ne rows', N(Fn)[hN==1][:2], 'Fx', N(Fx)[:2], sep='\n')
import pandas as pd
tr = pd.read_csv('/data/train.csv', nrows=5); print(list(tr.columns)); print(tr.head(3).to_string()[:400])
ks0 = np.where(sid3==sid3[0])[0]
out = run_stay_z(ks0, Eall, Uall)
print(type(out), [getattr(x,'shape',type(x)) for x in (out if isinstance(out,(tuple,list)) else [out])])
```