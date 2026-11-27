`tf_nll` returns a torch tensor, and `np.mean` doesn't accept it, so the run failed before scoring anything. The rest is sound: train is `tr`, tuning is `tu`, and sel1/sel2 hold 100 stays each. I'll score again using the tensor's own `.mean()`. In the same run I'll call `rollout_stay` on one tuning stay and print only the type and shape of its output, so I know how to compute shifts. As before, a build counts as viable if its factual cost is under about 0.003.

```python
def nll_of(Pd):
    r=tf_nll(Pd); return float(r.mean()) if hasattr(r,'mean') else float(r)
t=time.time(); n0=nll_of(PO); print('PO %.5f'%n0, round(time.time()-t,1),'s')
for nm in ['P1','P2']:
    print(nm, '%.5f  d=%+.5f'%(nll_of(globals()[nm]), nll_of(globals()[nm])-n0))
s0=S['sel1'][0]
try:
    t=time.time(); R=rollout_stay(tu, s0, PO, 4, 0); print('rollout type',type(R), round(time.time()-t,2),'s')
    if isinstance(R,dict): print({k:(getattr(v,'shape',None) or (len(v) if hasattr(v,'__len__') else v)) for k,v in R.items()})
    elif isinstance(R,(tuple,list)): print([getattr(x,'shape',type(x)) for x in R])
except Exception as ex: print('rollout err', str(ex)[:200])
```