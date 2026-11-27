E and the prelude FFN hidden units are now stored for every bundle and dose variant (16 s).

Next I will repeat the held-out ridge test with three feature sets: E alone (192 per slot), the prelude FFN hidden units (768), and both together. For each I report held-out R² of the insulin and norepinephrine targets at decision bundles, and the leakage RMS at other bundles.

**Outcomes:**
- High R² from E (for example above 0.7) with small leakage would confirm the `b_in` route.
- If only the prelude hidden units work, I would write through the prelude FFN instead and need its downstream response.

```python
def build(key):
    Xs_,Ys_,F_,Dm_=[],[],[],[]
    for d in D:
        f=d['sid'] in FIT
        if isdec(d):
            ri,rn,ci,cn=d['ri'],d['rn'],d['cI'],d['cN']
            L=[(d[key],ri-ci,rn-cn)]
            if 'hid_i2' in d: L+=[(d[key+'_i2'],2*ri-ci,rn-cn),(d[key+'_ih'],.5*ri-ci,rn-cn),(d[key+'_n2'],ri-ci,2*rn-cn),(d[key+'_nh'],ri-ci,.5*rn-cn)]
            for x_,a_,b_ in L: Xs_.append(x_); Ys_+= [[a_,b_]]*4; F_+=[f]*4; Dm_+=[1]*4
        else: Xs_.append(d[key]); Ys_+=[[0.,0.]]*4; F_+=[f]*4; Dm_+=[0]*4
    return torch.cat(Xs_).double(),torch.tensor(Ys_,dtype=torch.float64),torch.tensor(F_),torch.tensor(Dm_)
def evalfit(X_,Y_,F_,Dm_,lams):
    XtX=X_[F_].T@X_[F_]; XtY=X_[F_].T@Y_[F_]; n=X_.shape[1]; best=None
    for lam in lams:
        Rr=torch.linalg.solve(XtX+lam*torch.eye(n,dtype=torch.float64),XtY); Pv=X_@Rr
        v=~F_; dv=v&(Dm_==1); nv=v&(Dm_==0)
        r2s=[1-float(((Pv[dv,j]-Y_[dv,j])**2).mean()/Y_[dv,j].var()) for j in (0,1)]
        leak=[float(Pv[nv,j].pow(2).mean().sqrt()) for j in (0,1)]
        print('  lam %g val R2 I %.3f N %.3f | leak I %.3f N %.3f'%((lam,)+tuple(r2s)+tuple(leak)))
XE,YE,FE,DE=build('E'); print('E',XE.shape); evalfit(XE,YE,FE,DE,[1e-3,1e-2,1e-1,1])
XP,YP,FP,DP=build('ph'); print('ph',XP.shape); evalfit(XP,YP,FP,DP,[1e-2,1e-1,1,10])
```