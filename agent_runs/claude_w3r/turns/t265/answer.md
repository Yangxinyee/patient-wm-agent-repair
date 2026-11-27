Rate 0 does not mean "absent". `n1u(r, 0.0)` differs from the recorded single-drug tokens (max error 0.30 and 1.72), so the encoder must be getting a zero-rate atom for the other drug. `rg` was also never defined, because the previous run crashed before reaching it. I will call `u_of` directly with `None` for the missing drug, check that this reproduces the recorded single-drug tokens, and then run the linearity test.

```python
def n1x(a,b): return np.asarray(n1_of(np.asarray(u_of(a,b),dtype=float).reshape(-1)),dtype=float).reshape(-1)
print('None=absent check', np.abs(n1x(float(Yi[j1]),None)-Xu[j1]).max(), np.abs(n1x(None,float(Yn[j2]))-Xu[j2]).max())
rg=np.r_[0.25,0.5,1,1.5,2,3,4,5,6,7,8,9,10]
Gi=np.stack([n1x(float(r),None) for r in rg]); Gn=np.stack([n1x(None,float(r)/4) for r in rg])
for nm,yv_,G,sc in [('ins',Yi,Gi,1),('ne',Yn,Gn,4)]:
    for lamn in [0.1,1,10]:
        w=rread(Xu,yv_,Xn_,lamn)
        print(nm,lamn,'res',np.round((Xu@w-yv_).std(),3),'negmax',np.round(np.abs(Xoth@w).max(),3),'u0',np.round((u0@w)[0],3),'grid',np.round(G@w*sc,2))
```