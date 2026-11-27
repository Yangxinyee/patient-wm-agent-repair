The kernel holds three insulin builds (P_, P1, P2). Each changes 53 tensors and uses reserved dims jU=4 (staging from the dose token), jR=2 (reference) and jI=67 (memory). That matches the plan in my notes; these came from build_ins / build_ins2. P_ is only the last dict the head evaluator used, so the candidates are P1 and P2. I don't know how they performed, so I'll score both offline with the validated replica:
- factual NLL on the sel2 readings, against the original's 1.40449;
- the format of rollout_stay's output on one tuning stay, so I can compute dose-edit shifts next.

A candidate is worth keeping if it costs less than about 0.003 nats and moves the insulin ×2 shift well beyond the original −1.9, toward my estimated truth of −14.3.

```python
import time, pandas as pd
dfs=[(k,v.shape) for k,v in globals().items() if isinstance(v,pd.DataFrame) and not k.startswith('_')]
print('dfs',dfs[:10])
S=json.load(open('/work/sel.json')); print({k:len(v) for k,v in S.items()})
t=time.time(); b0=tf_nll(PO); print('PO', float(np.mean(b0)) if not isinstance(b0,float) else b0, round(time.time()-t,1),'s')
for nm in ['P1','P2']:
    r=tf_nll(globals()[nm]); print(nm, float(np.mean(r)) if not isinstance(r,float) else r)
```