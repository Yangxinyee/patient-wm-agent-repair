The z-draw idea is wrong. The across-draw variance of every readout is negligible (0.0009 against a between-row variance of 0.74 for insulin). The noise is between rows: the unit-gain insulin readout has noise variance 0.35 on top of a dev1 signal variance of 0.38.

The more likely reason the tool's cost is about twice my offline estimate is that the readouts were scored on the same stays they were fit on. Out of sample they are noisier.

So my offline cost numbers are only reliable if I fit readouts on some batches and score the factual cost on held-out ones. To set that up I need the exact return format of head_run. I'll run it with the original weights on dataset 0 and compare with base_out[0], timing it too.

```python
import time
ORIGd = {k: (ORIG[k] if isinstance(ORIG[k], np.ndarray) else npv(ORIG[k])) for k in ['W0','b0','W2','TR']}
t0 = time.time()
o = head_run(datasets[0][0], datasets[0][1], ORIG)
print('head_run secs %.1f' % (time.time()-t0), type(o), len(o) if isinstance(o,(tuple,list)) else '')
for j, part in enumerate(o if isinstance(o,(tuple,list)) else [o]):
    print(j, type(part).__name__, np.shape(part))
for j, part in enumerate(base_out[0]):
    print('base', j, type(part).__name__, np.shape(part))
print('pm match %.2e  nll match %.2e' % (np.abs(np.asarray(o[0])-np.asarray(base_out[0][0])).max(), np.abs(np.asarray(o[1])-np.asarray(base_out[0][1])).max()))
```