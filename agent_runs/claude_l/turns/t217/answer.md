`run_batch(W, stays)` returns a dict per stay, and `compare(R, f1, label)` presumably turns that into readouts set against the tool's original-model dose_edit output from forward 1.

Next I'll run the full batch-1 rollout (100 stays, 20 paths, all four edits) with the κ = 4 weights and pass it to `compare` with the forward-1 CSV. This gives the replica's four readouts for the edited model, side by side with the original model's tool readouts:
- insulin ×2: −3.25
- insulin ×½: +1.77
- NE ×2: +0.62
- NE ×½: −0.82

**Expectation:** κ = 4 should strengthen both drugs' effects noticeably, but probably to well below the record-based truth (insulin ×2 about −13, NE ×2 about +5). The size of the increase per unit κ will set the calibration.

```python
import pandas as pd
t0=time.time()
s0=keys[0]; print('inner keys', list(run_batch(Wk,[s0])[s0].keys())[:8])
R_k4=run_batch(Wk, keys)
f1=pd.read_csv('/work/fwd_1/dose_edit_original.csv')
print('rollout %.0fs'%(time.time()-t0))
try: compare(R_k4, f1, 'k4')
except Exception as ex: print('compare err', repr(ex)[:300])
```