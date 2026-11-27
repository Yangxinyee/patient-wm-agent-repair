The kernel still holds the decoders vI and vN, the replica edit responses (EDR), the head inputs (Xf, Rf) and the head weights. The original model's readouts on setA were −1.45, +0.82, +0.15 and −0.41. My record-based targets are −14.17, +7.09, +4.82 and −2.41. Next I will confirm what units the decoders use: their response to each simulated edit and their spread on the factual rows. That tells me whether the planned value-head correction can reach the targets within the factual budget. If the own response is about 1 per edit and the factual spread is small next to the gain needed, the plan holds.

```python
import inspect, numpy as np
s=inspect.getsource(solve_unit); print(s[:1200])
print(inspect.signature(head_mean))
for k,v in EDR.items():
    print(k, {kk:getattr(vv,'shape',type(vv).__name__) for kk,vv in v.items()})
try:
    for k,v in EDR.items():
        gi=v['gi']; dX=v['dX']; r=v['reached']
        code=cf[gi]
        for nm,vec,cs in [('vI',vI,[0,1]),('vN',vN,[2,3])]:
            m=np.isin(code,cs)&(r>0)
            print(k,nm,'resp',np.round((dX[m]@vec).mean(),3),'n',int(m.sum()))
except Exception as e: print('err',e)
for nm,vec,cs in [('vI',vI,[0]),('vN',vN,[2])]:
    m=np.isin(cf,cs); print(nm,'sdX',np.round((Xf[m]@vec).std(),3),'sdR',np.round((Rf[m]@vec).std(),3))
```