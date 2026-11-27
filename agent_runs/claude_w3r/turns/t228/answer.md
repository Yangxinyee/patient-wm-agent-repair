The quiet directions exist. The e, slot and z tokens after norm1 have many directions with sd near 0.003. Along those directions they all sit at the same constant as the empty u token, while real u tokens vary by 0.08–0.95. So u already carries information where the other tokens are nearly silent. My earlier fits failed only because they were confined to head 2's existing 48-dimensional value space.

Plan: add a rank-1 term to head 2's value weights that reads a quiet-space functional ψ of each token. Then ψ·n1(u) gives the rate at action bundles and a near-constant everywhere else. Next I will fit ψ for insulin and norepinephrine within the K quietest directions. Then I will check the actual stage-1 signal, which is head 2's attention-weighted Σ a_j ψ·n1_j. Success means held-out residual sd well below 0.1 at action bundles and noise below about 0.02 elsewhere.

```python
ins_s=PD['ins_s'].values.astype(float); ne_s=PD['ne_s'].values.astype(float)
def fit_psi(K,y,lam=1e-4):
    Bq=evec[:,1:K+1]                      # quiet NEG directions
    Xu=(n1[:,5]-mu)@Bq                     # u tokens, all bundles
    tr=act&~te
    Xa=np.c_[Xu[tr]]; G=Xa.T@Xa+lam*len(Xa)*np.eye(K); c=np.linalg.solve(G,Xa.T@(y[tr]))
    psi=Bq@c
    s=np.einsum('bij,bj->bi',patt[:,2,:4,:],(n1-mu)@psi)   # head2 attention-weighted read per slot
    r=s-y[:,None]
    return psi,[r[act&te].std(), np.abs(r[act&te]).max(), r[~act&te].std(), r[~act&te].mean(), ((NEG-mu)@psi).std(), np.linalg.norm(psi)]
for K in [10,20,40,60,90]:
    for nm,y in [('ins',ins_s),('ne',ne_s)]:
        psi,st=fit_psi(K,y); print(K,nm,'act sd %.3f max %.2f | noact sd %.4f mean %.4f | NEG sd %.4f | |psi| %.1f'%tuple(st))
```