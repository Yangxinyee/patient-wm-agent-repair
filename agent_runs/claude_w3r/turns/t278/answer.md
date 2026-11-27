The constrained linear reads stay poor: residual 0.32–0.47, and the grid is non-monotone. Halving a rate of 1 would even move the read the wrong way, so an ungated linear read is unusable. My notes say an action-only fit is accurate (sd 0.05), and that a dose-presence read is clean (leakage 0.07).

Plan: a gated channel. Head 2's lowest-energy value dims carry (gate, insulin rate, norepinephrine rate). A pair of dead units in the prelude FFN computes gate × rate, which removes the leakage at no-action bundles.

This run checks two things. First, whether a nearly unconstrained rate read is accurate and linear on the dose grid, including doubled and halved rates. Second, how large its no-action garbage is (that sets the gate margin). If the grid is linear, the gated design is viable. If it saturates or bends, I need a different read.

```python
anyp=np.ones(len(U_act))
def rr3(y,lamn,lam=1e-7,yneg=0.0):
    A=U_act.T@U_act/len(U_act)+lamn*(NEG.T@NEG/len(NEG)+np.outer(b1n,b1n))+lam*np.eye(192)
    return np.linalg.solve(A,U_act.T@y/len(U_act))
w_g=rr3(anyp,1.0,1e-6); print('gate res',round(float(np.std(U_act@w_g-1)),4),'NEG sd/max',round(float((NEG@w_g).std()),4),round(float(np.abs(NEG@w_g).max()),3),'u0',round(float(b1n@w_g),4))
gi=[.25,.5,.75,1,1.5,2,3,4,5,6,8,10]; gn=[.03,.1,.25,.5,.75,1,1.5,2]
for lamn in [0,1e-4,1e-3,1e-2]:
    for j,nm,gr in [(0,'ins',gi),(1,'ne',gn)]:
        w=rr3(pa[:,j],lamn); lk=NEG@w
        rg=[float(n1_of(u_of(**{nm:r}))@w) for r in gr]
        print(lamn,nm,'res',round(float(np.std(U_act@w-pa[:,j])),3),'leak sd',round(float(lk.std()),2),'max',round(float(np.abs(lk).max()),1),'u0',round(float(b1n@w),2),'grid',np.round(rg,2))
print('data rate values ins', np.unique(np.round(pa[pa[:,0]>0,0],2))[:20], 'ne', np.unique(np.round(pa[pa[:,1]>0,1],2))[:20])
```