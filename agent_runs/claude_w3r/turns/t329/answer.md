The loop filtered everything out — my guess that the action token sits at `EU[k][3]` must be wrong (no exception printed means the `continue` guards fired). I need the actual structure of an `EU` element before I can inject a doubled rate.

Next run inspects `EU` entries and the meaning of `GR['prow']`, including which entries are 192-vectors and which bundles are action bundles. That tells me the right index for the action token.

Confirming outcome: I identify a 192-dim entry in `EU[k]` that the insulin-rate probe reads with a sensible rate at known insulin bundles.

```python
import numpy as np
B,rws,EU=stay_data(int(GR['stay'].iloc[0]))
print('stay',int(GR['stay'].iloc[0]),'len EU',len(EU),'rows',rws[:12])
print(GR.head(6).to_string())
for k in range(min(8,len(EU))):
    e=EU[k]; print('EU[%d] type %s len %s'%(k,type(e).__name__,len(e) if hasattr(e,'__len__') else '-'))
    if isinstance(e,(tuple,list)):
        for j,x in enumerate(e):
            sh=getattr(x,'shape',None); nm=type(x).__name__
            extra=''
            if sh is not None and tuple(sh)==(192,):
                v=np.asarray(x.detach().cpu().numpy() if hasattr(x,'detach') else x,dtype=np.float64)
                extra=' |v|=%.2f probe_ins=%.2f probe_ne=%.2f'%(np.linalg.norm(v),float(pi.predict(v.reshape(1,-1))[0]),float(pn.predict(v.reshape(1,-1))[0]))
            elif sh is not None and len(tuple(sh))==2 and tuple(sh)[1]==192:
                V=np.asarray(x.detach().cpu().numpy() if hasattr(x,'detach') else x,dtype=np.float64)
                extra=' rows=%d probe_ins=%s'%(V.shape[0],np.round(pi.predict(V),2)[:5])
            print('   [%d] %s %s%s'%(j,nm,sh,extra), '' if sh is not None else repr(x)[:60])
print('UX sample |u| %.2f ; YXi>0 frac %.2f'%(np.linalg.norm(UX[0]),float((YXi>0).mean())))
```