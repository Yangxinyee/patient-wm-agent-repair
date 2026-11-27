The variance-reduced norepinephrine analysis gives the same weak picture as before:
- **Immediate effect:** the clearest direct estimate, lag 1, is +0.11 ± 0.08 mmHg per unit rate. So norepinephrine does essentially nothing right away.
- **Overall implied ×2 readout:** −0.24 ± 0.14, a 1.7-sigma signal.
- **By period:** before hour 12 it is −0.32; in the evaluation window (hours ≥ 12) it is +0.02.
- **Pattern:** the later-lag coefficients drift negative early in the stay and positive late in it, which looks like noise rather than a stable effect.

I conclude that norepinephrine most likely has no effect, and that 0 is a near-optimal estimate for its readouts. The dose-edit output shows that the original model's insulin ×2 shifts are small but systematic (for example +0.25 at hour 13 with se 0.009). So the original has a small spurious response that the dose-invariant edit removes.

Next I will apply the same cross-fitted method to insulin, using finger glucose as the outcome. I will also add the cuff readings as a second norepinephrine outcome: they share the true MAP but have independent measurement noise, so they add power.

- **Absence confirmed:** the insulin readout near 0 with a small standard error, and cuff agreeing with arterial at about 0.
- **Absence refuted:** a clearly non-zero insulin estimate, or cuff showing the same negative effect in hours ≥ 12.

```python
t0=time.time()
def lp_analysis(trt0, trt_nan, rowmask, Xfeats, outcome, Aw, K=12, nboot=200, label=''):
    si_,ti_=np.nonzero(rowmask)
    X_=np.stack([Xfeats[n][si_,ti_] for n in Xfeats],1)
    fold_=(stay_ids[si_]%2).astype(int)
    def xf(y):
        pred=np.full(len(y),np.nan); ok=~np.isnan(y)
        for f in (0,1):
            tr=ok&(fold_!=f); te=ok&(fold_==f)
            m=HGB(max_iter=200,learning_rate=0.1,max_leaf_nodes=31,min_samples_leaf=100).fit(X_[tr],y[tr]); pred[te]=m.predict(X_[te])
        return pred
    d=trt0[si_,ti_]; ep=d-xf(d)
    Lr={};Rr={}
    for k in range(1,K+1):
        tt=ti_+k; ok=tt<=25; y=np.full(len(si_),np.nan); y[ok]=outcome[si_[ok],tt[ok]]; Lr[k]=y-xf(y)
    for j in range(K):
        tt=ti_+j; ok=tt<=25; y=np.full(len(si_),np.nan); y[ok]=trt0[si_[ok],tt[ok]]; Rr[j]=y-xf(y)
    def cf(idx):
        e_=ep[idx]; L_=[];R_=[]
        for k in range(1,K+1):
            y=Lr[k][idx]; ok=~np.isnan(y); L_.append((e_[ok]*y[ok]).sum()/(e_[ok]**2).sum())
        for j in range(K):
            y=Rr[j][idx]; ok=~np.isnan(y); R_.append((e_[ok]*y[ok]).sum()/(e_[ok]**2).sum())
        return np.array(L_),np.array(R_)
    L_,R_=cf(np.arange(len(si_))); e_=deconv(L_,R_); ro=(e_*Aw).sum()
    us=np.unique(si_); order=np.argsort(si_); sis=si_[order]; st=np.searchsorted(sis,us); en=np.searchsorted(sis,us,side='right')
    rng=np.random.default_rng(1); bv=[]
    for b in range(nboot):
        pick=rng.integers(0,len(us),len(us)); idx=np.concatenate([order[st[p]:en[p]] for p in pick])
        Lb,Rb=cf(idx); bv.append((deconv(Lb,Rb)*Aw).sum())
    print(label,'n',len(si_),'innov sd %.3f'%ep.std(),'L',L_.round(2),'R',R_[:4].round(2),'readout %.3f boot sd %.3f'%(ro,np.std(bv)))
    out={}
    for nm,msk in [('t<12',(ti_-1)<12),('t>=12',(ti_-1)>=12)]:
        idx=np.nonzero(msk)[0]; Lm,Rm=cf(idx)
        bvm=[]
        for b in range(100):
            sub=idx[np.isin(si_[idx], us[rng.integers(0,len(us),len(us))])]  # approx
            if len(sub)<100: continue
            Lb,Rb=cf(sub); bvm.append((deconv(Lb,Rb)*Aw).sum())
        print('   ',nm,'readout %.3f (approx sd %.3f)'%((deconv(Lm,Rm)*Aw).sum(), np.std(bvm)))
    return ep, L_, R_
# NE with cuff outcome (same rows/features as before)
_=lp_analysis(NE0, NE, cont, feats, GC, Aw, label='NE->cuff')
# insulin
Gpol=np.where(np.isnan(GF),GL,GF)
ifeats={'gf0':GF,'gl0':GL,'gp0':Gpol,'gf1':lag(GF,1),'gl1':lag(GL,1),'gf2':lag(GF,2),'i1':lag(INS0,1),'i2':lag(INS0,2),
        'hr':hourmat,'cmF':cummean(GF),'cmL':cummean(GL)}
irows=~np.isnan(Gpol); irows[:,:3]=False; irows[:,25]=False
AwI=np.zeros(K); nI=0
for h in range(13,25):
    hi=h+1; ran=(INS0[:,13:hi]>0).any(1); rm=ran&~np.isnan(GF[:,hi]); nI+=rm.sum()
    for k in range(1,K+1):
        s=h-k
        if s>=12: AwI[k-1]+=INS0[rm,s+1].sum()
AwI/=nI; print('insulin eval-like rows',nI,'A(k)',AwI.round(2))
_=lp_analysis(INS0, INS, irows, ifeats, GF, AwI, label='INS->finger')
print('time',round(time.time()-t0,1))
```