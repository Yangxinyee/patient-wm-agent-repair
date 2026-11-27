That read failed: weighting the non-u tokens heavily ruined the fit, with residual 0.5 and a non-monotone grid. Run 245's reads (residual 0.13, linear over 0.5–5) must have used a lighter or normalised constraint on non-u tokens.

I will use the kernel's own `ridge_read` (whatever it returns) and scan the constraint weight. For each weight I check three things:
- the action-bundle residual,
- the leakage on non-u tokens, weighted as at no-action bundles where head 2 attends e at 0.92,
- linearity over the rate grid.

My earlier estimate was that leakage of about 0.2 rate units costs only about 0.001 nats, so an ungated channel may suffice.

```python
r0=ridge_read(U_act,pa[:,0],NEG); print('ridge_read returns',type(r0), getattr(r0,'shape',None) if not isinstance(r0,tuple) else [getattr(t,'shape',t) for t in r0])
def rr2(y,lamn,lam=1e-3):
    A=U_act.T@U_act/len(U_act)+lamn*(NEG.T@NEG/len(NEG)+np.outer(b1n,b1n))+lam*np.eye(192)
    return np.linalg.solve(A,U_act.T@y/len(U_act))
gi=[.5,1,1.5,2,3,4,5,6,8,10]; gn=[.03,.1,.25,.5,.75,1,1.5,2]
for lamn in [0.1,1,10]:
    for j,nm,gr in [(0,'ins',gi),(1,'ne',gn)]:
        w=rr2(pa[:,j],lamn); lk=NEG@w
        rg=[float(n1_of(u_of(**{nm:r}))@w) for r in gr]
        print(lamn,nm,'res',round(float(np.std(U_act@w-pa[:,j])),3),'leak sd',round(float(lk.std()),3),'max',round(float(np.abs(lk).max()),2),'u0',round(float(b1n@w),3),'grid',np.round(rg,2))
```