I'll keep this strictly technical. The edit in place is the gated three-unit z writer (units 12, 88, 105), with gains -0.18 and 0.6. `hG`/`hM` are not impulse responses, so I need to find where my per-lag effect kernels are stored (likely `KAP`) and which loaded tables hold the records. Then I can pick tuning stays for a dose_edit check and compute the target on the same stays. If `KAP` holds per-lag kernels, I can compute targets right away. If not, I'll re-estimate them from the training records.

```python
import pandas as pd
print({k:(type(v).__name__, (list(v.keys())[:10] if isinstance(v,dict) else None)) for k,v in [('KAP',KAP)]})
for k,v in list(KAP.items())[:8]:
    try: print(k, np.round(np.asarray(v,dtype=float).ravel()[:8],3))
    except Exception: print(k, type(v).__name__)
dfs=[(k,v.shape,list(v.columns)[:9]) for k,v in globals().items() if isinstance(v,pd.DataFrame)]
for d in dfs: print(d)
```