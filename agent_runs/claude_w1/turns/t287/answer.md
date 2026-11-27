The four decoder responses scale correctly: insulin ×2 gives +1.60 and ×1/2 gives −0.80, norepinephrine gives +0.52 and −0.25, and there is no cross-leak. But the factual spread of the decoder outputs (cell-residualised sd 14.2 and 2.9) is far larger than these responses. A correction built on them could hurt the factual fit badly. I need to know whether that spread is heavy-tailed outliers or real noise, and what one decoder unit means in dose terms. If the robust spread is small, the plan is viable; if not, I must redesign the decoders.

```python
import numpy as np
for k,v in EDR.items():
    r=np.asarray(v['reached'])>0; print(k,'mean dd lags',np.round(v['dd'][r].mean(0),3))
for nm,vec,cs in [('vI',vI,[0]),('vN',vN,[2])]:
    m=np.isin(cf,cs); u=Rf[m]@vec
    q=np.percentile(u,[1,5,25,50,75,95,99]); mad=np.median(np.abs(u-np.median(u)))*1.4826
    print(nm,'pct',np.round(q,2),'robust sd',np.round(mad,3))
cand=[n for n,v in list(globals().items()) if isinstance(v,np.ndarray) and v.ndim>=1 and v.shape[0]==70738]
print('70738-row arrays:',cand[:30])
```