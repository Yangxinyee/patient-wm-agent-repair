The kernel cannot show function source code. Instead I will print solve_unit's argument names, the structure of EDR, and the decoder responses and spreads directly. This tells me the decoder units before I build the edit.

```python
import numpy as np
c=solve_unit.__code__; print('solve_unit args',c.co_varnames[:c.co_argcount],'locals',c.co_varnames[c.co_argcount:c.co_argcount+12])
c=head_mean.__code__; print('head_mean args',c.co_varnames[:c.co_argcount])
for k,v in EDR.items():
    print(k, {kk:getattr(vv,'shape',type(vv).__name__) for kk,vv in v.items()})
try:
    for k,v in EDR.items():
        gi=v['gi']; dX=v['dX']; r=np.asarray(v['reached'])
        code=cf[gi]
        for nm,vec,cs in [('vI',vI,[0,1]),('vN',vN,[2,3])]:
            m=np.isin(code,cs)&(r>0)
            print(k,nm,'resp',np.round((dX[m]@vec).mean(),3),'n',int(m.sum()))
except Exception as e: print('err',repr(e)[:200])
for nm,vec,cs in [('vI',vI,[0]),('vN',vN,[2])]:
    m=np.isin(cf,cs); print(nm,'sdX',np.round((Xf[m]@vec).std(),3),'sdR',np.round((Rf[m]@vec).std(),3),'norm',np.round(np.linalg.norm(vec),4))
```