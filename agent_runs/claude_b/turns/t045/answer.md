Splitting the residual resolves the contradiction:

- **Clean intensity noise (A):** the multiplicative noise on given doses at readings ≥150 is essentially exogenous. Its distributed-lag effect is null (χ²(12)=8.7), with an implied insulin x2 shift of +0.76 ± 1.06 mg/dL.
- **The other two components are strongly endogenous:**
  - (B) The "omissions" depend on the previous reading.
  - (C) Doses given at readings below 150 or with no reading depend strongly on the previous reading and previous dose.
  
  These two produced the spurious large positive associations, including the +1.4 single-lag estimate.

So the clean natural experiment still says insulin has no effect, but the sustained-effect bound is loose: SE about 1 mg/dL.

The norepinephrine residual I used mixes clean titration noise with possibly endogenous stops and starts. Its null could hide a real effect cancelled by confounding, the way B and C distorted the insulin estimate.

Next run:
1. Tighten the insulin A estimate with richer pre-window controls: earlier readings, labs, stay running mean and earlier doses.
2. Split norepinephrine into three parts:
   - NA: ratio-based titration noise;
   - NB: stops;
   - NS: starts.
3. Test NA's exogeneity and run the distributed-lag regression on it.
4. Estimate how long a titration shock persists in the rate.

What would settle it:
- If both clean components stay null with tighter SEs, zero effects (my current edit) are confirmed.
- A significant clean NA effect would mean norepinephrine does act and the edit must change.

```python
def dl_multi(Y, Elist, names, fe_fns, cont_fn, drug, label, tmin=13, tmax=24):
    Ys=[];Ls=[];FEs=[[] for _ in fe_fns];Cs=[];Ts=[];SS=[];DR=[]
    for t in range(tmin,tmax+1):
        ti=t+1; y=Y[:,ti]; ok=~np.isnan(y)
        Ys.append(y[ok]); Ls.append(np.hstack([np.stack([E[ok,ti-k] for k in range(1,13)],1) for E in Elist]))
        for j,f in enumerate(fe_fns): FEs[j].append(f(ti)[ok])
        Cs.append(cont_fn(ti)[ok]); Ts.append(np.full(ok.sum(),t)); SS.append(np.where(ok)[0]); DR.append(np.stack([drug[ok,ti-k] for k in range(1,13)],1))
    y=np.concatenate(Ys); L=np.concatenate(Ls); C=np.concatenate(Cs); tt=np.concatenate(Ts); st=np.concatenate(SS); D=np.concatenate(DR)
    fes=[np.concatenate(F) for F in FEs]+[tt]
    b,V=cl_ols(y,np.hstack([L,C]),fes,st)
    print(label,'N',len(y))
    kmask=(np.arange(1,13)[None,:]<=(tt[:,None]-12)).astype(float); CC=D*kmask; reached=(CC>0).any(1); c=CC[reached].mean(0)
    out={}
    for j,nm in enumerate(names):
        bj=b[12*j:12*j+12]; Vj=V[12*j:12*j+12,12*j:12*j+12]
        print('  %s lags:'%nm,' '.join('%+.2f(%.2f)'%(bj[i],np.sqrt(Vj[i,i])) for i in range(12)))
        print('  %s chi2(12)=%.1f sum=%+.3f(%.3f) implied x2 = %+.3f (se %.3f)'%(nm,bj@np.linalg.solve(Vj,bj),bj.sum(),np.sqrt(Vj.sum()),c@bj,np.sqrt(c@Vj@c)))
        out[nm]=(bj,Vj,c)
    return out
def fill(a): return np.nan_to_num(a), np.isnan(a).astype(float)
cs=np.cumsum(np.nan_to_num(Fg),1); cc=np.cumsum(~np.isnan(Fg),1); runF=np.where(cc>0,cs/np.maximum(cc,1),np.nan)
def cont_ins(ti):
    cols=[]
    for a in [Fg[:,ti-12],Fg[:,ti-13],Fg[:,ti-14],Lb[:,ti-12],Lb[:,ti-13],runF[:,ti-12]]:
        v,m=fill(a); cols+= [v,m]
    cols+=[Ins[:,ti-13],Ins[:,ti-14]]
    return np.stack(cols,1)
oI=dl_multi(Fg,[eA],['insA'],[lambda ti: fbc[:,ti-12]],cont_ins,Ins,'INSULIN(A) rich controls')
oI2=dl_multi(Lb,[eA],['insA->lab'],[lambda ti: fbc[:,ti-12]],cont_ins,Ins,'INSULIN(A) -> lab')
# NE components
prevN=Ne[:,:-1]; curN=Ne[:,1:]; cont_=prevN>0; on=cont_&(curN>0)
cellN=(mb[:,1:]+1)*30+np.clip(np.floor(prevN/0.05),0,20).astype(int)
ratio=np.where(on,curN/np.maximum(prevN,1e-9),0)
r_mean=np.bincount(cellN[on],weights=ratio[on],minlength=1800)/np.maximum(np.bincount(cellN[on],minlength=1800),1)
p_on=np.bincount(cellN[cont_],weights=on[cont_].astype(float),minlength=1800)/np.maximum(np.bincount(cellN[cont_],minlength=1800),1)
startc=(prevN==0)&(curN>0); st_mean=np.bincount((mb[:,1:]+1)[prevN==0],weights=curN[prevN==0],minlength=60)/np.maximum(np.bincount((mb[:,1:]+1)[prevN==0],minlength=60),1)
NA=np.zeros_like(Ne); NB=np.zeros_like(Ne); NS=np.zeros_like(Ne)
NA[:,1:]=np.where(on,prevN*(ratio-r_mean[cellN]),0); NB[:,1:]=np.where(cont_,(on-p_on[cellN])*prevN,0); NS[:,1:]=np.where(prevN==0,curN-st_mean[mb[:,1:]+1],0)
for nm,E in [('NA',NA),('NB',NB),('NS',NS)]:
    ys=[];x1=[];f1=[];ss=[]
    for s in range(2,25):
        si=s+1; ok=(cont_[:,si-1] if nm!='NS' else (prevN[:,si-1]==0)) & ~np.isnan(Mu[:,si-1])
        ys.append(E[ok,si]); x1.append(np.stack([Mu[ok,si-1]/10, np.nan_to_num(Ar[ok,si]-Cf[ok,si]), np.isnan(Ar[ok,si]-Cf[ok,si]), Ne[ok,si-1]],1)); f1.append(cellN[ok,si-1]); ss.append(np.where(ok)[0])
    b,V=cl_ols(np.concatenate(ys),np.concatenate(x1),[np.concatenate(f1)],np.concatenate(ss))
    print(nm,'sd %.4f exog: prevMAP/10 %+.4f(%.4f) art-cuff %+.5f(%.5f) prevprevRate %+.4f(%.4f)'%(E[:,1:].std(),b[0],np.sqrt(V[0,0]),b[1],np.sqrt(V[1,1]),b[3],np.sqrt(V[3,3])))
rbN=np.clip(np.floor(Ne/0.05),0,20).astype(int)
csA=np.cumsum(np.nan_to_num(Ar),1); ccA=np.cumsum(~np.isnan(Ar),1); runA=np.where(ccA>0,csA/np.maximum(ccA,1),np.nan)
def cont_ne(ti):
    cols=[]
    for a in [Ar[:,ti-12],Ar[:,ti-13],Ar[:,ti-14],Cf[:,ti-12],runA[:,ti-12]]:
        v,m=fill(a); cols+=[v,m]
    cols+=[Ne[:,ti-13],Ne[:,ti-14]]
    return np.stack(cols,1)
oN=dl_multi(Ar,[NA,NB,NS],['NA','NB','NS'],[lambda ti: mbc2[:,ti-12], lambda ti: rbN[:,ti-13]],cont_ne,Ne,'NE components -> art MAP')
# persistence of NA shock in rate
pers=[]
for j in range(0,12):
    ys=[];xs=[];f1=[];ss=[]
    for s in range(1,25-j):
        si=s+1; ok=cont_[:,si-1]
        ys.append(Ne[ok,si+j]); xs.append(NA[ok,si]); f1.append(cellN[ok,si-1]); ss.append(np.where(ok)[0])
    b,V=cl_ols(np.concatenate(ys),np.concatenate(xs)[:,None],[np.concatenate(f1)],np.concatenate(ss)); pers.append(b[0])
print('NA persistence in rate, lags 0..11:',np.round(pers,3))
```