Run 63 corrects my reading of run 62. The reading-bundle z vectors have no exact null space: their singular values fall smoothly to about 0.15, so the R² of "1.000" was only very high, not perfect. The low-variance directions are still usable. Along them, reading bundles vary about 20 times less than drug bundles, and the result held on held-out stays.

A plain symmetric rank1 along that direction would, however, push z mostly along other dimensions, not along z26. What I want is a non-symmetric update of `z_in.weight`, W ← W(I + α·e26·vᵀ):
- read a functional v·z that is close to 0 at reading bundles and equals z26 − m_t at drug bundles of type t;
- write the result along the model's own z26 input direction.

That amplifies the z26 deviation at drug bundles (1+α)-fold around the mean of each bundle type, and leaves reading bundles nearly unchanged. Such an update is a product of two SPD matrices, so four symmetric rank1 'in' ops can build it.

Now I will fit v by constrained least squares, check how much it leaks at reading bundles, both in-sample and held out, and verify the 4-op decomposition numerically.

```python
def fit_v(Zread, cts, mts, lam=1e4):
    # v = e26 + w, w26=0; minimize ||Zread v||^2 + lam*sum_t (w.c_t + m_t)^2
    others=[j for j in range(32) if j!=26]
    A=Zread[:,others]; y=-Zread[:,26]
    Ac=np.array([c[others] for c in cts])*np.sqrt(lam); yc=-np.array(mts)*np.sqrt(lam)
    AA=np.vstack([A,Ac]); yy=np.concatenate([y,yc])
    w,_,_,_=np.linalg.lstsq(AA,yy,rcond=None)
    v=np.zeros(32); v[26]=1; v[others]=w
    return v
types=['ins_only','ne_only','both']
cts=[zo_[grp_[t]].mean(0) for t in types]; mts=[zo_[grp_[t],26].mean() for t in types]
v_=fit_v(Zr,cts,mts)
print('|v| %.3f, max|w| %.3f'%(np.linalg.norm(v_),np.abs(np.delete(v_,26)).max()))
fr=Zr@v_
print('reading f: mean %.4f sd %.4f max|f| %.3f  (z26 read sd %.3f)'%(fr.mean(),fr.std(),np.abs(fr).max(),Zr[:,26].std()))
for t in types:
    m=grp_[t]; f=zo_[m]@v_; dev=zo_[m,26]-zo_[m,26].mean()
    print(t,'f vs z26-m: corr %.4f, resid sd %.4f, f mean %.4f'%(np.corrcoef(f,dev)[0,1],(f-dev).std(),f.mean()))
# held-out: fit on even stays, test on odd stays
ev=(sid_%2==0)
cts_e=[zo_[grp_[t]&ev].mean(0) for t in types]; mts_e=[zo_[grp_[t]&ev,26].mean() for t in types]
v_e=fit_v(zo_[mr&ev],cts_e,mts_e)
fo=zo_[mr&~ev]@v_e
print('HELD-OUT reading f: mean %.4f sd %.4f max|f| %.3f'%(fo.mean(),fo.std(),np.abs(fo).max()))
for t in types:
    m=grp_[t]&~ev; f=zo_[m]@v_e; dev=zo_[m,26]-mts_e[types.index(t)]
    print('  HO',t,'resid sd %.4f mean %.4f'%((f-dev).std(),(f-dev).mean()))
# 4-op decomposition of M = I + alpha e26 v^T
def decomp_ops(a,b,alpha):
    q1=a/np.linalg.norm(a); bb=b-(b@q1)*q1; q2=bb/np.linalg.norm(bb); Q=np.column_stack([q1,q2])
    M2=np.eye(2)+alpha*np.outer(Q.T@a,Q.T@b)
    ev_,P=np.linalg.eig(M2); ev_=ev_.real; P=P.real
    S1=P@np.diag(ev_)@P.T; S2=np.linalg.inv(P@P.T)
    out=[]
    for S in (S1,S2):
        lam_,E=np.linalg.eigh(S)
        for k in range(2):
            out.append((Q@E[:,k], lam_[k]-1))
    return out
alpha=4.0
dops=decomp_ops(e26,v_,alpha)
Wt=np.random.randn(192,32); Wc=Wt.copy()
for vec,al in dops:
    vec=vec/np.linalg.norm(vec); Wc=Wc+al*np.outer(Wc@vec,vec)
print('decomp err', np.abs(Wc-Wt@(np.eye(32)+alpha*np.outer(e26,v_))).max(), [round(a,3) for _,a in dops])
```