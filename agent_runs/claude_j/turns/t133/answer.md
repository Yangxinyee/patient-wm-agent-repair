`reading_table(S, BUN)` gives all 5,214 value readings of the 100 tuning stays, with the state row before each reading (`prow`), the time gap, and the last-value features. That's exactly what the head needs. Next I'll compute the head queries q for these readings (`q_for`), the dose-history targets (`hourly_maps` + `add_targets`), and load the tool's factual predictions from fwd_2 to merge residuals. Only shapes and column names are printed. That confirms the pieces line up before I fit anything.

```python
import pandas as pd, numpy as np
Xz=np.asarray(S['x']); Q=q_for(Xz, RTz); print('Q', getattr(Q,'shape',None), type(Q).__name__)
tus=tu[tu.stay_id.isin(list(BUN.keys()))]
HM=hourly_maps(tus); print('hourly_maps ->', type(HM).__name__, len(HM) if hasattr(HM,'__len__') else '', [type(x).__name__ for x in HM] if isinstance(HM,tuple) else '')
RTt=add_targets(RTz.copy(), *HM); print('new cols', [c_ for c_ in RTt.columns if c_ not in RTz.columns])
fac=pd.read_csv('/work/fwd_2/factual_original.csv'); print('fac', fac.shape, list(fac.columns))
```