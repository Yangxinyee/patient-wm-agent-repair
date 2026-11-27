Run 64 shows that the selective functional exists but is costly to build. v has norm 42, because it cancels against other z dimensions with coefficients up to 17. Its leakage at held-out reading bundles is sd 0.22, against z26's natural sd of 1.57, and at α=4 that would disturb reading bundles noticeably. The plain two-SPD factorization was also badly conditioned, with alphas of -1.0 and 3509, which is risky if the tool applies ops in float32.

Next I will map the trade-off with a ridge penalty on the cancelling coefficients:
- how |v| moves against held-out leakage at reading bundles and fidelity at drug bundles;
- a better-conditioned factorization, using the free eigenvector scaling to balance the two SPD factors;
- a float32 simulation of applying the ops in sequence, to measure the rounding error.

A ridge setting with |v| of about 5 or less and held-out leakage below 0.1 would make a clean experiment possible.

```python
def fit_v_ridge(Zread, cts, mts, lam=1e4, ridge=0.0):
    others=[j for j in range(32) if j!=26]
    A=Zread[:,others]; y=-Zread[:,26]
    Ac=np.array([c[others] for c in cts])*np.sqrt(lam); yc=-np.array(mts)*np.sqrt(lam)
    AA=np.vstack([A,Ac,np.sqrt(ridge)*np.eye(31)]); yy=np.concatenate([y,yc,np.zeros(31)])
    w,_,_,_=np.linalg.lstsq(AA,yy,rcond=None)
    v=np.zeros(32); v[26]=1; v[others]=w
    return v
print('ridge | |v| | HO read leak mean, sd | HO drug resid sd (ins,ne,both)')
for rg in [0,1,10,30,100,300,1000,3000]:
    ve=fit_v_ridge(zo_[mr&ev],cts_e,mts_e,ridge=rg)
    fo=zo_[mr&~ev]@ve
    rs=[]
    for t in types:
        m=grp_[t]&~ev; f=zo_[m]@ve; dev=zo_[m,26]-mts_e[types.index(t)]
        rs.append((f-dev).std())
    print(rg, round(np.linalg.norm(ve),2), round(fo.mean(),4), round(fo.std(),4), np.round(rs,3))
def decomp_balanced(a,b,alpha,ngrid=401):
    q1=a/np.linalg.norm(a); bb=b-(b@q1)*q1; q2=bb/np.linalg.norm(bb); Q=np.column_stack([q1,q2])
    M2=np.eye(2)+alpha*np.outer(Q.T@a,Q.T@b)
    ev_,P=np.linalg.eig(M2); ev_=ev_.real; P=P.real
    best=None
    for lg in np.linspace(-8,8,ngrid):
        Pl=P@np.diag([1,np.exp(lg)])
        S1=Pl@np.diag(ev_)@Pl.T; S2=np.linalg.inv(Pl@Pl.T)
        c=max(np.linalg.cond(S1),np.linalg.cond(S2))
        if best is None or c<best[0]: best=(c,S1,S2)
    c,S1,S2=best
    out=[]
    for S in (S1,S2):
        lam_,E=np.linalg.eigh(S)
        for k in range(2): out.append((Q@E[:,k], lam_[k]-1))
    return out,c
def sim32(W,dops):
    Wc=W.astype(np.float32)
    for vec,al in dops:
        vec=(vec/np.linalg.norm(vec)).astype(np.float32)
        Wc=Wc+np.float32(al)*np.outer(Wc@vec,vec).astype(np.float32)
    return Wc
for rg in [0,30,300]:
    vv_=fit_v_ridge(Zr,cts,mts,ridge=rg)
    dops,c=decomp_balanced(e26,vv_,4.0)
    Wt=np.random.randn(192,32)*0.2
    exact=Wt@(np.eye(32)+4.0*np.outer(e26,vv_))
    err=np.abs(sim32(Wt,dops)-exact).max()/np.abs(exact).max()
    print('ridge',rg,'|v| %.2f cond %.1f alphas'%(np.linalg.norm(vv_),c),[round(a,3) for _,a in dops],'f32 rel err %.2e'%err)
```