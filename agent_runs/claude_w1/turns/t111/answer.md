The panels hold what I need. `Ptr3` has the NE policy variables and the NE innovation `nuN` (rate minus policy-expected rate), and `Ptr2` has the glucose value the insulin policy used (`Gany`), the insulin dose `D` and `prev_on`.

Next I'll generate factual (unperturbed) head-input samples with `gen_samples(pert=False)`, timing it on 40 training stays first. I need them for the factual covariance of the head input, which sets the decoder's noise floor. I'll also check that the output has the same tuple layout as `samp_pert`.

```python
import time
stays_all = np.array(sorted(tr.stay_id.unique())); used = set(ints.tolist())
rng = np.random.default_rng(7); cand = np.array([s for s in rng.choice(stays_all, 400, replace=False) if s not in used])[:40]
t0 = time.time(); fs = gen_samples(tr[tr.stay_id.isin(cand)], pert=False); t1 = time.time()
print('time %.1fs for %d stays, rows %d' % (t1-t0, len(cand), len(fs)), 'tuple len', len(fs[0]), [type(v).__name__ for v in fs[0]])
print('row ex:', fs[3][:8])
```