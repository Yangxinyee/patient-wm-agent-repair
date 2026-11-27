Stage 1 fails as designed. The constrained linear reads of the rate from the normalised action token have errors of 0.21–0.59 on real tokens. On the synthetic grid the response is non-monotonic at low rates and flat above about 4–5 (the insulin read stays near 4.3 for every rate from 5 to 10). Doubled rates (up to about 8) would therefore be badly underread.

I need to know whether the bending comes from the token itself or from the constraint against other tokens. Earlier notes claimed a linear rate probe on the raw action vector held even for out-of-range doubled rates. Next I fit unconstrained reads on the raw action vector and on the normalised token and evaluate both on the 0.25–10 grid. If either stays linear up to 10, the rate is recoverable and the constraint is the problem. If both bend, the token simply does not encode rates above the data range.

```python
Ugi=np.stack([np.asarray(u_of(float(r),None),dtype=float).reshape(-1) for r in rg])
Ugn=np.stack([np.asarray(u_of(None,float(r)/4),dtype=float).reshape(-1) for r in rg])
URAW=np.stack([np.asarray(u_of(float(Yi[j]) if Yi[j]>0 else None, float(Yn[j]) if Yn[j]>0 else None),dtype=float).reshape(-1) for j in range(len(Yi))])
def ridge1(X,y,lam=1e-3):
    X1=np.c_[X,np.ones(len(X))]; wb=np.linalg.solve(X1.T@X1+lam*np.eye(X1.shape[1]),X1.T@y); return wb
for nm,yv_,Gn1,Gr,sc in [('ins',Yi,Gi,Ugi,1),('ne',Yn,Gn,Ugn,4)]:
    for lab,X,G in [('n1',Xu,Gn1),('raw',URAW,Gr)]:
        for lam in [1e-3,1e-1]:
            wb=ridge1(X,yv_,lam); p=X@wb[:-1]+wb[-1]
            print(nm,lab,lam,'res',np.round((p-yv_).std(),3),'grid',np.round((G@wb[:-1]+wb[-1])*sc,2))
print('rate range data ins',np.percentile(Yi[Yi>0],[1,50,99]).round(2),'ne',np.percentile(Yn[Yn>0],[1,50,99]).round(2))
```