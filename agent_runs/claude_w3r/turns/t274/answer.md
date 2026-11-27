Head 2's `out` block has no near-null singular values (its minimum is 0.30), so the "free value dims" in my notes cannot be null directions of the weights. They must be directions in head 2's 48-dim value space that the data never excites. If so, a new rate read can occupy those directions, and the `out` block can be redirected along them, without changing the original computation on real tokens.

This run will:
- confirm those directions from the token values (`U_act`, `NEG`, `u0`),
- fit clean insulin and norepinephrine rate reads that are near zero on non-u tokens,
- check the reads' linearity, including the doubled out-of-range rates.

Confirmation would be at least 2 value directions with near-zero energy and rate reads with residual ≈0.13 that are roughly linear up to the doubled rates.

```python
b1n=load('core.prelude.0.norm1.bias').double().numpy()
X0=np.vstack([U_act,NEG,b1n[None]]); VV=X0@V2.T
ev,evec=np.linalg.eigh(VV.T@VV/len(VV)); print('value energy low', np.round(ev[:5],12), 'next', np.round(ev[5:8],6))
NF=evec[:,:3]
def rr(y,lamn=1.0,lam=1e-3,wu0=300.):
    A=U_act.T@U_act+lamn*NEG.T@NEG+wu0*np.outer(b1n,b1n)+lam*len(U_act)*np.eye(192)
    return np.linalg.solve(A,U_act.T@y)
for lam in [1e-4,1e-3]:
    w_i=rr(pa[:,0],lam=lam); w_n=rr(pa[:,1],lam=lam)
    gi=[.25,.5,1,1.5,2,3,4,5,6,8,10]; gn=[.03,.1,.25,.5,.75,1,1.25,1.5,2]
    ri=[float(n1_of(u_of(ins=r))@w_i) for r in gi]; rn=[float(n1_of(u_of(ne=r))@w_n) for r in gn]
    print(lam,'ins res',np.std(U_act@w_i-pa[:,0]).round(3),'NEGmax',np.abs(NEG@w_i).max().round(3),'u0',float(b1n@w_i).round(3),'grid',np.round(ri,2))
    print(lam,'ne  res',np.std(U_act@w_n-pa[:,1]).round(3),'NEGmax',np.abs(NEG@w_n).max().round(3),'u0',float(b1n@w_n).round(3),'grid',np.round(rn,2))
    print(' cross: ins-read on ne-only', np.round([float(n1_of(u_of(ne=r))@w_i) for r in [.5,1,2]],2), 'ne-read on ins-only', np.round([float(n1_of(u_of(ins=r))@w_n) for r in [1,3,6]],2))
```