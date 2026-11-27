The last run's output was cut at 10,000 characters, mostly by the function list. So I still have not seen the README, the dose-edit section of ARCHITECTURE.md, or the per-hour profile of fwd_9.

The function list confirms the reusable pieces I need are in the kernel:
- `build_edit(kI, kN, ...)` rebuilds the edit;
- `mk_fb2(rate_side, dose_side)` and `fbin(..., side)` handle bin conventions;
- `scale_atoms` applies dose edits;
- `DEC_I`/`DEC_N` (residual decoders) and `CH_I_new`/`CH_N_new` (c_hat ridges) are the fitted pieces.

Next I print only the parts I need, in compact form:
- the README description of drug rows;
- the ARCHITECTURE.md text around "dose edit";
- the per-hour mean shifts in fwd_9 for all four edits;
- the per-stay spread at h13.

If insulin ×½ is near zero or positive at h13 and turns negative with each later bundle, the encoding artifact builds up at every edited drug bundle. If instead a few stays dominate, it is tied to specific rate values or bins.

```python
import re, pandas as pd, numpy as np
R = open('/data/README.md').read()
print(R[:1600]); print('-----')
A = open('/model/ARCHITECTURE.md').read()
idx = [m.start() for m in re.finditer(r'(?i)dose[_ ]?edit|counterfactual|what.if', A)]
print('ARCH len', len(A), 'hits', idx[:8])
if idx:
    s = max(0, idx[0]-200); print(A[s:s+2200])
print('-----')
d = pd.read_csv('/work/fwd_9/dose_edit_edited.csv')
print(d.columns.tolist()); print(d.head(2).to_string()[:500])
ec = next((c for c in d.columns if 'edit' in c.lower()), None)
hc = next((c for c in d.columns if c.lower() in ('hour','stay_hour')), None) or next((c for c in d.columns if 'hour' in c.lower()), None)
sc = next((c for c in d.columns if 'shift' in c.lower()), None)
rc = next((c for c in d.columns if 'reach' in c.lower()), None)
dd = d[d[rc].astype(bool)] if rc else d
try:
    print(dd.groupby([ec, hc])[sc].mean().unstack(0).round(2).to_string())
    h0 = dd[hc].min()
    p = dd[dd[hc] == h0].pivot_table(index='stay_id', columns=ec, values=sc)
    print('first hour', h0, 'quantiles:\n', p.quantile([.1,.25,.5,.75,.9]).round(2).to_string())
    print('frac<0:', (p < 0).mean().round(2).to_dict())
except Exception as e:
    print('fail', repr(e))
```