Each `res_tf` variant is a list of 4,989 tuples (stay, time, code, predicted mean, NLL), aligned with `cur`, the original model's teacher-forced results on the 100 tuning stays of batch 1. So I can compare the variants directly against the baseline.

`pi_i` and `pi_n` are linear decodes of the expected dose from the context state (a 192-vector plus a bias). `w_ins` and `w_ne` are probably the token-space injection directions. `cf` holds the per-unit-rate directions of u, with norms of about 10.

Next I'll compute, for `dup`, `ki4` and `ki8`:
- the mean NLL change against `cur`, overall and per code;
- the spread of the predicted-mean shift.

I'll also try to read back the source of my helper `lagreg`, so I can reuse it for the residual-response regression.

What to expect:
- `dup` should show a change near 0 (around 1e-4 or less) if copying u into the z token is harmless.
- `ki4`/`ki8` should show a negative NLL change on glucose if amplifying the residual moves the model toward the truth. The records already showed the model under-responds to residuals, so a correct amplification should help the factual fit slightly.

```python
import inspect
base=np.array([r[4] for r in cur]); codes=np.array([r[2] for r in cur])
for k,v in res_tf.items():
    nl=np.array([r[4] for r in v]); mu=np.array([r[3] for r in v]); mu0=np.array([r[3] for r in cur])
    d=nl-base; s=mu-mu0
    print(k,'dNLL %.5f'%d.mean(), {int(c):round(float(d[codes==c].mean()),4) for c in np.unique(codes)}, 'shift sd %.3f maxabs %.2f'%(s.std(),np.abs(s).max()))
try: print(inspect.getsource(lagreg)[:1500])
except Exception as ex: print('no src',ex, inspect.signature(lagreg))
```