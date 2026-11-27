Fixes and new problems from that run:

- **Insulin, rich controls:** the regression was numerically singular, which is my bug. The missing-reading indicator duplicates the "missing" bin of the fixed effect. I'll switch to a pseudo-inverse.
- **Insulin, lab outcome:** null again, implied x2 +0.62 ± 1.39.
- **Norepinephrine titration noise (NA) is not perfectly clean.** It depends on the previous hour's MAP (−0.003 per 10 mmHg, 6σ) and on the previous rate within a rate bin (the cap truncates upward steps). Controlling only at the start of the window then leaves a small bias, probably negative and a few tenths of a mmHg per lag. That could explain the slightly negative norepinephrine lags seen earlier.

The norepinephrine regression also crashed on a singular matrix.

Next run uses single-lag impulse responses, controlled at the dose time, which removes that bias:

- **Insulin:** dose-time controls are the reading-bin fixed effect plus the current, previous and lab readings and the previous dose.
- **Norepinephrine:** dose-time controls are the MAP-bin × previous-rate fixed effect, absorbed by demeaning, plus the previous MAP, current arterial and cuff readings, the rate two hours back and hour.
- **Rate persistence:** I'll estimate how long a dose shock lasts in later doses.
- **Converting to x2 shifts:** for norepinephrine I'll deconvolve the MAP responses by that persistence to get per-hour direct effects. Summing direct effects over the recorded doses then gives the implied x2 shift.

What would settle it:
- Implied x2 shifts near zero with SEs well under 1 confirm the zero-effect edit.
- A clearly non-zero clean effect refutes it.

```python
def demean(A, g, ng):
    A=np.asarray(A,float); 
    if A.ndim==1: A=A[:,None]
    cnt=np.bincount(g,minlength=ng).astype(float)
    M=np.stack([np.bincount(g,weights=A[:,j],minlength=ng) for j in range(A.shape[1])],1)/np.maximum(cnt,1)[:,None]
    return A-M[g]
def fe_ols(y, Xr, cell, small_fes, st):
    N=len(y); ug,g=np.unique(cell,return_inverse=True); ng=len(ug)
    cols=[Xr]
    for f in small_fes:
        u,ii=np.unique(f,return_inverse=True); D=np.zeros((N,len(u)-1)); m=ii>0; D[np.where(m)[0],ii[m]-1]=1; cols.append(D)
    X=demean(np.hstack(cols),g,ng); yd=demean(y,g,ng)[:,0]
    XtX=X.T@X; Bi=np.linalg.pinv(XtX); b=Bi@(X.T@yd); u=yd-X@b
    G=np.zeros((n,X.shape[1])); np.add.at(G,st,X*u[:,None]); V=Bi@(G.T@G)@Bi
    p=Xr.shape[1]; return b[:p],V[:p,:p]
def impulse(Y, E, okfn, cellfn, ctrlfn, K=12, label=''):
    bs=[];ses=[];per_obs=[]
    for k in range(0,K+1):
        ys=[];xs=[];cl=[];cs=[];hh=[];ss=[]
        for s in range(1,25-k):
            si=s+1; ok=okfn(si)&~np.isnan(Y[:,si+k])
            ys.append(Y[ok,si+k]); xs.append(E[ok,si]); cl.append(cellfn(si)[ok]); cs.append(ctrlfn(si)[ok]); hh.append(np.full(ok.sum(),s)); ss.append(np.where(ok)[0])
        y=np.concatenate(ys); x=np.concatenate(xs)[:,None]; C=np.concatenate(cs)
        b,V=fe_ols(y,np.hstack([x,C]),np.concatenate(cl),[np.concatenate(hh)],np.concatenate(ss))
        bs.append(b[0]); ses.append(np.sqrt(V[0,0]))
    print(label,' '.join('k%d:%+.3f(%.3f)'%(k,bs[k],ses[k]) for k in range(K+1)))
    return np.array(bs),np.array(ses)
def ctrl_ins(si):
    cols=[]
    for a in [Fg[:,si],Fg[:,si-1],Lb[:,si],runF[:,si]]:
        v,m=fill(a); cols+=[v,m]
    cols+=[Ins[:,si-1]]; return np.stack(cols,1)
okI=lambda si: hi[:,si]
bI1,sI1=impulse(Fg,eA,okI,lambda si: fb[:,si],ctrl_ins,label='INS A -> finger')
pI,_=impulse(Ins,eA,okI,lambda si: fb[:,si],ctrl_ins,K=11,label='INS A -> later insulin dose (persistence)')
def ctrl_ne(si):
    cols=[]
    for a in [Mu[:,si-1],Ar[:,si],Cf[:,si],runA[:,si]]:
        v,m=fill(a); cols+=[v,m]
    cols+=[Ne[:,si-2] if si>=2 else np.zeros(n), Ne[:,si-1]]; return np.stack(cols,1)
okN=lambda si: cont_[:,si-1]
cellfnN=lambda si: cellN[:,si-1]
bN1,sN1=impulse(Ar,NA,okN,cellfnN,ctrl_ne,label='NE NA -> art MAP')
pN,_=impulse(Ne,NA,okN,cellfnN,ctrl_ne,K=11,label='NE NA -> later NE rate (persistence)')
```