This result contradicts my notebook's conclusion. With the dose residuals at the 12 preceding hours entered together, insulin residuals predict higher later finger glucose. The coefficients grow with lag, from +0.33 at lag 1 to +1.5 at lag 12, jointly very significant (χ² 182), implying an x2 shift of about +7 mg/dL. For norepinephrine the coefficients are jointly null (χ² 9.4, implied −1.0 ± 0.65).

Either insulin has a real, slowly building glucose-raising effect, or part of my "residual" is not random.

The prime suspect is a mixed residual. It combines three components:
- the multiplicative noise on given doses, already shown to be iid;
- the ~3% omitted doses;
- doses given when the reading is below 150 (p≈0.29), which may depend on history, e.g. continuing an infusion.

Next run:
1. Replicate the old single-lag regression, which controlled for the reading at the dose hour.
2. Split the residual into these three components.
3. Test each component for dependence on the recent past: previous reading and previous dose.

What would settle it:
- If the positive effect lives only in a component that depends on the past, it is confounding and the null stands.
- If the clean multiplicative noise alone shows the growing positive effect, the effect is real.

```python
def cl_ols(y, Xr, fes, st):
    # OLS with dummy FEs (list of int arrays), cluster by stay st; returns beta, V for Xr columns
    N=len(y); cols=[Xr]
    for f in fes:
        u,ii=np.unique(f,return_inverse=True); D=np.zeros((N,len(u)-1)); m=ii>0; D[np.where(m)[0],ii[m]-1]=1; cols.append(D)
    X=np.hstack(cols+[np.ones((N,1))]); XtX=X.T@X; b=np.linalg.solve(XtX,X.T@y); u=y-X@b
    G=np.zeros((n,X.shape[1])); np.add.at(G,st,X*u[:,None]); Bi=np.linalg.inv(XtX); V=Bi@(G.T@G)@Bi
    p=Xr.shape[1]; return b[:p], V[:p,:p]
S_=np.repeat(np.arange(n)[:,None],H,1)
# 1. single-lag replication with FE on reading bin at dose time
for k in [1,2,3,6,9,12]:
    ys=[];xs=[];f1=[];f2=[];ss=[]
    for s in range(0,25-k):
        si=s+1; y=Fg[:,si+k]; ok=~np.isnan(y)
        ys.append(y[ok]); xs.append(Eins[ok,si]); f1.append(fbc[ok,si]); f2.append(np.full(ok.sum(),s)); ss.append(np.where(ok)[0])
    b,V=cl_ols(np.concatenate(ys),np.concatenate(xs)[:,None],[np.concatenate(f1),np.concatenate(f2)],np.concatenate(ss))
    print('single-lag k=%d: %+.3f (%.3f)'%(k,b[0],np.sqrt(V[0,0])))
# 2. components
given=Ins>0; hi=(fb>=15)
cnt_g=np.bincount((fb+1)[given&hi],minlength=62); sm_g=np.bincount((fb+1)[given&hi],weights=Ins[given&hi],minlength=62); mg=sm_g/np.maximum(cnt_g,1)
eA=np.where(given&hi, Ins-mg[fb+1],0.0)
pg=np.bincount((fb+1)[hi],weights=given[hi].astype(float),minlength=62)/np.maximum(np.bincount((fb+1)[hi],minlength=62),1)
eB=np.where(hi,(given.astype(float)-pg[fb+1])*mg[fb+1],0.0)
lo=~hi
eC=np.where(lo,Ins-cm[fb+1],0.0)
eA[:,0]=eB[:,0]=eC[:,0]=0
for nm,E in [('A intensity',eA),('B omission',eB),('C low/missing-reading doses',eC)]:
    print(nm,'var share: sd %.3f, nonzero frac %.3f'%(E[:,1:].std(), (np.abs(E[:,1:])>1e-9).mean()))
    # exogeneity: regress E_s on prev reading & prev dose, FE on current bin
    ys=[];x1=[];f1=[];ss=[]
    for s in range(2,25):
        si=s+1; ok=(np.abs(E[:,si])>1e-9)|(hi[:,si] if nm[0]!='C' else lo[:,si])
        ok&=~np.isnan(Fg[:,si-1])
        ys.append(E[ok,si]); x1.append(np.stack([Fg[ok,si-1]/100, Ins[ok,si-1], Lb[ok,si-1]*0 if True else 0],1)[:,:2]); f1.append(fbc[ok,si]); ss.append(np.where(ok)[0])
    b,V=cl_ols(np.concatenate(ys),np.concatenate(x1),[np.concatenate(f1)],np.concatenate(ss))
    print('  exog test: coef prevFinger/100 %+.4f (%.4f), prevDose %+.4f (%.4f)'%(b[0],np.sqrt(V[0,0]),b[1],np.sqrt(V[1,1])))
    bb,VV=dl_reg(Fg,E,fbc,None,Ins,'  DL '+nm)
```