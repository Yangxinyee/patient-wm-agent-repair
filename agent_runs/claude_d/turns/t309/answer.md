Run 279 is a clear negative for scaling beta. The ideal bin-16 representation moves u by 2.6 at beta=1 (r′=2), but only to 3.7 at beta=8 and about 3.9 at most. The layer norm in the encoder saturates, so a bigger beta can add at most about +0.2 on top of E5's +0.32. The run crashed before saving, so the current edit is still E5 (beta=1).

I also found that `build_enc(1.0)` with default settings does not reproduce the saved E5 gate weights. I won't trust it for rebuilds.

Before investing the remaining runs in a different NE mechanism (for example extrapolating along the direction in u that recorded rate increases actually move), I need to know whether the +4.85 target is right. About 80% of it (3.97) comes from extrapolating a linear-in-rate effect beyond the recorded cap of 1.0. If the records show the effect flattening at high rates, the true ne_x2 effect is smaller and the current +1.35 may not be far off.

This run looks for a training-set hourly panel to test curvature: what `PT_tr`, `TRAIN` and similar objects are, and the arguments of the panel helpers. It also prints the notebook lines on the effect estimate that were cut off last time.

```python
import json, numpy as np
txt = open('/work/notes.md').read()
L = [l for l in txt.split('\n') if any(k in l for k in ['linear','satur','Emax','gam_ne','curv','quadr','nonlin'])]
print('\n'.join(L)[:1500])
for k in ['PT_tr','PT','TRAIN','TR','TRd','TRe','TRi','TRn','panel','gl_all','glm_all','all_tr']:
    v_ = globals().get(k)
    if v_ is None: continue
    s = getattr(v_,'shape',None) or (len(v_) if hasattr(v_,'__len__') else None)
    c = list(v_.columns)[:30] if hasattr(v_,'columns') else (list(v_.keys())[:20] if isinstance(v_,dict) else '')
    print(k, type(v_).__name__, s, c)
for f_n in [hourly_panel, dose_panel, stay_panel, gest_sub, iv_fit, add_lags, lagD]:
    co = f_n.__code__; print(f_n.__name__, co.co_varnames[:co.co_argcount])
print(list(json.load(open('/work/effects.json')).keys()))
```