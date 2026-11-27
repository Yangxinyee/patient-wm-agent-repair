These magnitudes change my picture. The prelude attention output is large (|v| ≈ 120–275) and strongly dose-dependent. Insulin moves it by ~83 between 0.5 and 6 u/h (a saturating, roughly log-like shape, 87–91% along one direction). The per-state scale of h is ~47 per channel (CV 0.16).

There is also an NE problem: rates above 1.0 (fine bins never seen in training) move v by ~151. That is a huge, untrained jump, and the NE x2 edit hits it.

So the dose signal is strong at the source, yet only 0.043 per u/h reaches the head input. It is lost somewhere in between: in the prelude LayerNorm, the core update, the 55-minute time decay before the next reading, or the head's LayerNorm. Finding where tells me where an amplification can act on a clean signal. I'll check what `run_stay` returns, so I can trace a single dose perturbation stage by stage.

```python
Bq = bundles_of_stay(grp_tr[s_ins], RU)
t0 = time.time(); R0 = run_stay(Bq); print('run_stay %.2fs, type %s' % (time.time()-t0, type(R0).__name__))
if isinstance(R0, dict): print({k: (type(v).__name__, np.shape(v)) for k, v in R0.items()})
elif isinstance(R0, (list, tuple)):
    print('len', len(R0), type(R0[0]).__name__)
    r0 = R0[0]; print({k: (type(v).__name__, np.shape(v)) for k, v in r0.items()} if isinstance(r0, dict) else [np.shape(v) for v in r0] if isinstance(r0, (list, tuple)) else np.shape(r0))
```