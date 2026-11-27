This run removes the omissions explanation:
- **Insulin, jitter only:** using only rows where insulin was actually given, the readout is still +0.89 ± 0.48. The early/late split stays the same: hours < 12 give +0.06, hours ≥ 12 give +2.04. Omissions barely relate to the glucose trend.
- **Norepinephrine, combined MAP:** the combined MAP outcome gives a ×2 readout of −0.34 ± 0.14. That is −0.37 before hour 12 and −0.17 from hour 12 on.
- **Random-halves check:** it crashed because my "halves" hash preserves stay-id parity, the same rule that defines the cross-fitting folds, so each half held one fold only.

The insulin result is now the biggest open question. The late-window estimate looks like a 3–3.5 sigma signal after correcting my rough subset standard error. If real, it means insulin slightly raises glucose after hour 12 (about +2 mg/dL for ×2), and the original model misses it. Alternatives are a hidden transient confounder of the jitter, or noise.

Before believing either, I will look at how the effect varies with time in finer windows (hours 2–5, 6–9, 10–13, 14–17, 18–23). I will use per-lag coefficients with proper stay-clustered bootstrap standard errors, computed from one cross-fit. I will also cross-check with a deliberately simple estimator: future finger change on the log-jitter, with 5 mg/dL glucose-bin fixed effects.

- **Real time-varying effect:** a smooth trend with time in the per-lag effects, agreeing across both estimators.
- **Artifact or noise:** windows bouncing around 0, or the simple estimator disagreeing.

```python
t0=time.time()
def build_resid(trt0, rowmask, Xfeats, outcome, K=12):
    si_,ti_=np.nonzero(rowmask); X_=np.stack([Xfeats[n][si_,ti_] for n in Xfeats],1); fold_=((stay_ids[si_]//7)%2).astype(int)
    def xf(y):
        pred=np.full(len(y),np.nan); ok=~np.isnan(y)
        for f in (0,1):
            tr=ok&(fold_!=f); te=ok&(fold_==f)
            pred[te]=HGB(max_iter=200,max_leaf_nodes=31,min_samples_leaf=100).fit(X_[tr],y[tr]).predict(X_[te])
        return pred
    d=trt0[si_,ti_]; ep=d-xf(d); Ly=[];Ry=[]
    for k in range(1,K+1):
        tt=ti_+k; ok=tt<=25; y=np.full(len(si_),np.nan); y[ok]=outcome[si_[ok],tt[ok]]; Ly.append(y-xf(y))
    for j in range(K):
        tt=ti_+j; ok=tt<=25; y=np.full(len(si_),np.nan); y[ok]=trt0[si_[ok],tt[ok]]; Ry.append(y-xf(y))
    return dict(si=si_,ti=ti_,ep=ep,Ly=np.array(Ly),Ry=np.array(Ry))
def window_stats(Rz, msk, Aw, K=12, nb=300, seed=0):
    si_=Rz['si'][msk]; ep=Rz['ep'][msk]; Ly=Rz['Ly'][:,msk]; Ry=Rz['Ry'][:,msk]
    us,inv=np.unique(si_,return_inverse=True); ns=len(us)
    def persum(Y):
        ok=~np.isnan(Y); num=np.zeros((Y.shape[0],ns)); den=np.zeros((Y.shape[0],ns))
        for k in range(Y.shape[0]):
            np.add.at(num[k],inv[ok[k]],(ep*np.nan_to_num(Y[k]))[ok[k]]); np.add.at(den[k],inv[ok[k]],(ep**2)[ok[k]])
        return num,den
    nL,dL=persum(Ly); nR,dR=persum(Ry)
    def est(w):
        L=(nL@w)/(dL@w); R=(nR@w)/(dR@w); return L,R,(deconv(L,R)*Aw).sum()
    L,R,ro=est(np.ones(ns))
    rng=np.random.default_rng(seed); W=rng.multinomial(ns,np.ones(ns)/ns,size=nb).T.astype(float)
    Lb=(nL@W)/(dL@W); Rb=(nR@W)/(dR@W)
    rob=np.array([(deconv(Lb[:,b],Rb[:,b])*Aw).sum() for b in range(nb)])
    return L, Lb.std(1), ro, rob.std()
RI=build_resid(INS0, irows_pos, ifeats, GF)
print('ins resid built',round(time.time()-t0,1))
hr_=RI['ti']-1
for lo,hi in [(2,5),(6,9),(10,13),(14,17),(18,23),(2,11),(12,23)]:
    L,Ls,ro,ros=window_stats(RI,(hr_>=lo)&(hr_<=hi),AwI)
    print('INS t in [%d,%d]'%(lo,hi),'L1-6',L[:6].round(2),'se',Ls[:6].round(2),'readout %.2f (%.2f)'%(ro,ros))
# simple estimator: finger change on log-jitter with 5mg/dL glucose-bin FE, rows INS>0 with finger at t
si3,ti3=np.nonzero(irows_pos&~np.isnan(GF))
g=GF[si3,ti3]; b=base_of(g); jit=np.log(INS0[si3,ti3]/np.maximum(b,1e-9)); okb=b>0
print('jitter range',np.exp(np.percentile(jit[okb],[1,50,99])).round(2))
gb=(g//5).astype(int)
for k in (1,3,5,7):
    tt=ti3+k; ok=(tt<=25)&okb; y=np.full(len(si3),np.nan); y[ok]=GF[si3[ok],tt[ok]]-g[ok]; ok&=~np.isnan(y)
    for nm,wm in [('t<12',(ti3-1)<12),('t>=12',(ti3-1)>=12)]:
        m=ok&wm; df=pd.DataFrame({'y':y[m],'x':INS0[si3[m],ti3[m]],'gb':gb[m],'s':si3[m]})
        df['yd']=df.y-df.groupby('gb').y.transform('mean'); df['xd']=df.x-df.groupby('gb').x.transform('mean')
        beta=(df.xd*df.yd).sum()/(df.xd**2).sum(); u=df.xd*(df.yd-beta*df.xd); cl=u.groupby(df.s).sum()
        se=np.sqrt((cl**2).sum())/(df.xd**2).sum()
        print('  simple k=%d %s beta %.3f se %.3f n %d'%(k,nm,beta,se,m.sum()))
print('time',round(time.time()-t0,1))
```