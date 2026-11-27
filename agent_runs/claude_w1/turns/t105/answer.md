No function source is recoverable. `samp_pert` holds 39,178 tuples of the form (stay id, int, 6 floats, a 192-vector), probably one row per reading with a head-input vector and dose/innovation features. To use it I need to know what the columns are. I'll print a few rows and per-column summaries (range, mean, sd), plus the vector norm, and the overlap of the 192-vector with known head-query arrays. That should identify the columns, e.g. the reading value, lagged doses or innovations, and whether the vector is q or `with_code`.

```python
import numpy as np
ids = [t[0] for t in sp]; ints = np.array([t[1] for t in sp]); F = np.array([t[2:8] for t in sp]); V = np.stack([t[8] for t in sp])
print('unique stays', len(set(ids)), 'int range', ints.min(), ints.max(), 'int uniq', np.unique(ints)[:12])
np.set_printoptions(precision=3, suppress=True, linewidth=150)
print('first rows:'); print(np.column_stack([ints[:5], F[:5]]))
print('col mean', F.mean(0)); print('col sd  ', F.std(0)); print('col min ', F.min(0)); print('col max ', F.max(0))
print('vec norm mean/sd', np.linalg.norm(V,axis=1).mean(), np.linalg.norm(V,axis=1).std(), 'vec mean of means', V.mean(1).mean(), 'vec sd mean', V.std(1).mean())
print('corr among float cols'); print(np.corrcoef(F.T))
```