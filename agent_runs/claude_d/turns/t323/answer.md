Run 289 shows the problem clearly. On factual tuning rows, the edit's shifts barely point toward the observed values.

| Readings | dNLL | Shift rms | Slope of residual on shift | Note |
|---|---|---|---|---|
| map_arterial | +0.019 | 1.0 | 0.03 | NE channel; shifts are pure noise |
| glucose_finger | +0.012 | 2.7 | 0.22 | insulin channel; cost almost all on the 39% of rows with large shifts |
| glucose_lab | +0.001 | — | — | — |
| map_cuff | small gain | — | — | — |

As designed, the glucose slope should be about 0.6. A simple Gaussian calculation, shift variance × (1 − 2 × slope) / (2σ²), reproduces the measured costs. So the cost is decoder noise that does not carry over from the training-stay states to tuning stays.

Conclusions:
- The NE head channel must go. It buys about 1 to 2 mmHg of NE effect at about +0.0065 overall.
- The insulin channel is the remaining cost, about +0.004 overall on its own, which is too close to the 0.005 limit.

Next I will fit dNLL ≈ a·shift² + b·shift·residual per event. From that I can predict the factual cost of scaling the insulin channel's output by a factor f with NE removed, and get a stay-bootstrap standard error. I will also read fwd13, the sel1 dose_edit with E5, for reference.

How to read the result:
- If f = 1 comes out near +0.004 with SE about 0.001, full insulin gain is unsafe and I should scale down.
- If the predicted cost is well under 0.003 at a high f, I can keep most of the insulin effect.

```python
import json, numpy as np, pandas as pd
d=fo[['stay_id','hour','event','observed']].copy(); d['dn']=fe.nll-fo.nll; d['dp']=fe.predicted_mean-fo.predicted_mean; d['e0']=fo.observed-fo.predicted_mean
coef={}
for ev,g in d.groupby('event'):
    X=np.c_[g.dp**2, g.dp*g.e0]; c=np.linalg.lstsq(X,g.dn.values,rcond=None)[0]; coef[ev]=c
    r2=1-((g.dn-X@c)**2).sum()/((g.dn-g.dn.mean())**2).sum()
    print(ev[:9], 'a %.5f b %.5f r2 %.2f'%(c[0],c[1],r2))
N=len(d); gl=d.event.str.startswith('glucose').values
A=np.array([coef[e][0] for e in d.event]); B=np.array([coef[e][1] for e in d.event])
rng=np.random.default_rng(0); sid=d.stay_id.values; us=np.unique(sid); idx={s:np.where(sid==s)[0] for s in us}
for f in (0.5,0.6,0.7,0.8,0.9,1.0):
    r=np.where(gl, A*f*f*d.dp**2+B*f*d.dp*d.e0, 0.0)
    ss=np.array([r[idx[s]].sum() for s in us]); nn=np.array([len(idx[s]) for s in us])
    bs=[ss[i].sum()/nn[i].sum() for i in rng.integers(0,len(us),(1000,len(us)))]
    print('f %.1f est cost %.4f se %.4f'%(f, r.sum()/N, np.std(bs)))
print('empirical gluc part f=1 %.4f'%(np.where(gl,d.dn,0).sum()/N))
s13=json.load(open('/work/fwd_13/summary.json'))['results']
print('fwd13', {m:{k:round(v['mean_shift'],2) for k,v in s13[m]['edits'].items()} for m in s13})
```