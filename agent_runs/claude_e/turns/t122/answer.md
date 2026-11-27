`mid` is 0.46, which is the norepinephrine rate centre (z = (r − 0.46)/0.59). My print failed on `np.size` of a torch tensor, a bug in my inspection code. I will convert to numpy first and repeat the comparison of ED with `make_edit(1,1)` and `make_edit(2,2)`.

If ED's output columns are an exact multiple of the unit edit, the edit is linear in its coefficients and that multiple is the coefficient ED was built with.

```python
def npv(v):
    return v.detach().cpu().numpy() if hasattr(v,'detach') else np.asarray(v)
for nm in ['tg','tm','gvec','CENT']:
    if nm in globals():
        v=globals()[nm]
        if isinstance(v,(list,tuple)): print(nm,'list',len(v),[np.round(npv(x),2) for x in v][:4]); continue
        a=npv(v).astype(float); print(nm, a.shape, np.round(a,3) if a.size<=40 else ('norm %.3f'%np.linalg.norm(a)))
E1=make_edit(1.0,1.0); E2=make_edit(2.0,2.0)
for k in ['W0','b0','W2','TR']:
    print(k,'|E1-O|',float((E1[k]-ORIG[k]).abs().sum()),'|E2-O|',float((E2[k]-ORIG[k]).abs().sum()),'|ED-O|',float((ED[k]-ORIG[k]).abs().sum()))
for u in units:
    a1=(E1['W2'][:,u]-ORIG['W2'][:,u]); ae=(ED['W2'][:,u]-ORIG['W2'][:,u]); a2=(E2['W2'][:,u]-ORIG['W2'][:,u])
    r=float((ae*a1).sum()/(a1*a1).sum()); r2_=float((a2*a1).sum()/(a1*a1).sum())
    print('unit',u,'W2: ED/E1 %.3f E2/E1 %.3f resid %.4f'%(r,r2_,float((ae-r*a1).abs().max())),
          '| W0row ED-E1 max %.4f'%float((ED['W0'][u]-E1['W0'][u]).abs().max()),'| b0 ED %.3f E1 %.3f O %.3f'%(float(ED['b0'][u]),float(E1['b0'][u]),float(ORIG['b0'][u])))
```