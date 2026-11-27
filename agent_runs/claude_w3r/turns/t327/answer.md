The Hmean reader works but has a fatal signal-to-noise ratio: under a dose doubling the read moves only ~0.011, while its cross-sectional spread is 0.25 — a 23:1 ratio against us. So scaling it to reach −13.7 would need ~290x gain and would destroy the factual fit. Confirmed by the rollout: ins×2 went only −0.52 → −0.84.

I therefore need a reader direction chosen to maximise *intervention* sensitivity, not cross-sectional fit. I can measure that locally with the kernel's single-bundle simulator (`step_at` with a modified action token), with no forward passes. First I must confirm the helper signatures and the layout of the per-stay data so I can inject a doubled rate.

Confirming outcome: I can locate the action token inside `EU` and get the per-bundle rate, so the next run can compute Δq per unit rate for the original model.

```python
import inspect, numpy as np
G=globals()
for f in ['step_at','stay_data','run_from','u_of','wc_row','coda','apply_local','core_step_v']:
    if f in G:
        try: print(f, inspect.signature(G[f]))
        except Exception as e: print(f,'sig?',e)
    else: print(f,'MISSING')
s=int(selS[0]); print('stay',s)
out=stay_data(s); print('stay_data ->',type(out).__name__,len(out))
B,rows,EU=out
for nm,v in (('B',B),('rows',rows),('EU',EU)):
    print(nm,type(v).__name__,getattr(v,'shape',None),getattr(v,'dtype',None))
    if hasattr(v,'columns'): print('  cols',list(v.columns))
if hasattr(rows,'head'): print(rows.head(14).to_string())
elif isinstance(rows,(list,tuple)): print('rows[:6]',rows[:6])
if isinstance(EU,(list,tuple)):
    print('EU len',len(EU),[getattr(e,'shape',type(e).__name__) for e in EU[:6]])
r=step_at(B,rows,EU,3,None,None) if True else None
print('step_at ->',type(r).__name__, len(r) if hasattr(r,'__len__') else '')
if hasattr(r,'__len__'):
    for i,x in enumerate(r): print('  out',i,type(x).__name__,getattr(x,'shape',None))
print('u_of test', np.asarray(u_of(1.5,0.5)).shape, float(np.abs(np.asarray(u_of(1.5,0.5))).max()))
```