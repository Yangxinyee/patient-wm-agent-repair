The signatures give me the building blocks:
- `n1u(a, b)` returns the normalised action token for given drug rates.
- `prelude_att` and `step_at` run the validated local core step.
- `Xoth` (15,603 × 192) holds the normalised non-action tokens, `Xu` (2,406 × 192) the action tokens, and `Yi`/`Yn` their two drug rates.

Next I test the first stage of the channel: a bias-free linear read of each rate from the normalised action token. It is constrained to stay near zero on all other tokens, because the attention head's value has no bias. I will check its error on real action tokens, its leakage on other tokens, and whether it stays linear over a synthetic rate grid that covers the doubled range. If the read bends at high rates, the ×2 readout would be understated and I would need a nonlinear correction stage.

```python
import numpy as np
def rread(Xp,yp,Xn,lam_n=1.0,lam=1e-3):
    X=np.vstack([Xp,np.sqrt(lam_n)*Xn]); y=np.r_[yp,np.zeros(len(Xn))]
    return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@y)
chk=[np.abs(np.asarray(n1u(Yi[j] if Yi[j]>0 else None, Yn[j] if Yn[j]>0 else None)).reshape(-1)-Xu[j]).max() for j in range(3)]
print('Xu check',np.round(chk,5))
u0=np.asarray(n1u(None,None)).reshape(1,-1)
Xn_=np.vstack([Xoth,u0.repeat(200,0)])
rg=np.r_[0.25,0.5,1,1.5,2,3,4,5,6,7,8,9,10]
Gi=np.stack([np.asarray(n1u(float(r),None)).reshape(-1) for r in rg])
Gn=np.stack([np.asarray(n1u(None,float(r)/4)).reshape(-1) for r in rg])
for nm,yv_,G,sc in [('ins',Yi,Gi,1),('ne',Yn,Gn,4)]:
    for lamn in [0.1,1,10]:
        w=rread(Xu,yv_,Xn_,lamn)
        print(nm,lamn,'res',np.round((Xu@w-yv_).std(),3),'negmax',np.round(np.abs(Xoth@w).max(),3),'u0',np.round((u0@w)[0],3),
              'grid',np.round(G@w*sc,2))
```