Re-orienting after the restart. The last run checked that the counterfactual event function is the patched version and that one stay's with_code and logit tensors have the expected shapes. My recent replies were blocked, so from now on I print only aggregates and keep replies short.

Plan: check what the kernel still holds: the cf dataset M with WC/LG, the replica functions, and the reading counts per event. Then I can fit a dose readout without regenerating data. If M/WC/LG exist with the expected row counts, I continue with the head-unit plan. If not, I regenerate a smaller dataset.

```python
import numpy as np, pandas as pd, os, types
def shp(v):
    return getattr(v,'shape',None) if hasattr(v,'shape') else (len(v) if hasattr(v,'__len__') else '')
for n in ['M','WC','LG','RTs','BUN','W','Wt','CEN','vG','vM','SHIFT','Xt']:
    v=globals().get(n,None); print(n, type(v).__name__, shp(v) if v is not None else '-')
fn=[k for k,v in globals().items() if isinstance(v,types.FunctionType) and not k.startswith('_')]
print('FUNCS', ' '.join(sorted(fn))[:1500])
if isinstance(globals().get('M'),pd.DataFrame):
    print('Mcols', list(M.columns))
    for c in ['drug','h0','m','event']:
        if c in M.columns: print(c, sorted(M[c].astype(str).unique().tolist())[:12])
f2=pd.read_csv('/work/fwd_2/factual_original.csv'); print('fwd2 counts', f2['event'].value_counts().to_dict())
```