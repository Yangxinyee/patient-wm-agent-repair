The machine-learning-free cell estimator reproduces the norepinephrine pattern at lower strength. Its lags 1–2 are about 0 (+0.11, +0.02), it is negative from lag 3 (−0.14 to −0.30), and its implied ×2 readout is −0.22. The gradient-boosting version gave −0.34. So fitting bias is not creating the signal, but the size depends somewhat on method. Across methods the norepinephrine estimate is about −0.2 to −0.35.

The decision between zeros and installing small effects turns on how significant these robust, ML-free estimates are. I will:
- bootstrap the cell estimator's standard error for norepinephrine (combined and arterial-only outcomes; the readout uses arterial);
- build the matching ML-free cell estimator for insulin (5 mg/dL glucose cells, finger outcome) with its implied ×2 readout and bootstrap.

- **Favours "absent":** both ML-free readouts have p > 0.1. Zeros are then clearly preferable.
- **Favours installing shrunken effects:** both are clearly significant, and I would weigh engineering effects of about half the estimated size into the model.

```python
t0=time.time()
def cellfe_readout(sR,tR,trtR,cellR,outcome,trt0,Aw_,K=12,nb=300,diff=True):
    us,inv=np.unique(sR,return_inverse=True); ns=len(us)
    nL=np.zeros((K,ns)); dL=np.zeros((K,ns)); nR=np.zeros((K,ns)); dR=np.zeros((K,ns))
    def dm(v,ok):
        df=pd.DataFrame({'v':v[ok],'c':cellR[ok]}); return (df.v-df.groupby('c').v.transform('mean')).to_numpy()
    for k in range(1,K+1):
        tt=tR+k; ok=tt<=25; y=np.full(len(sR),np.nan); y[ok]=outcome[sR[ok],tt[ok]]-(outcome[sR[ok],tR[ok]] if diff else 0)
        ok=~np.isnan(y); yd=dm(y,ok); xd=dm(trtR,ok)
        np.add.at(nL[k-1],inv[ok],xd*yd); np.add.at(dL[k-1],inv[ok],xd*xd)
    for j in range(K):
        tt=tR+j; ok=tt<=25; y=np.full(len(sR),np.nan); y[ok]=trt0[sR[ok],tt[ok]]
        ok=~np.isnan(y); yd=dm(y,ok); xd=dm(trtR,ok)
        np.add.at(nR[j],inv[ok],xd*yd); np.add.at(dR[j],inv[ok],xd*xd)
    L=nL.sum(1)/dL.sum(1); R=nR.sum(1)/dR.sum(1); ro=(deconv(L,R)*Aw_).sum()
    rng=np.random.default_rng(5); W=rng.multinomial(ns,np.ones(ns)/ns,size=nb).T.astype(float)
    Lb=(nL@W)/(dL@W); Rb=(nR@W)/(dR@W); rob=np.array([(deconv(Lb[:,b],Rb[:,b])*Aw_).sum() for b in range(nb)])
    return ro, rob.std(), L
# NE cell-FE (rows with reading): combined and arterial outcomes
for nm,outc in [('combined',MAPc),('arterial',GA)]:
    ro,se,L=cellfe_readout(sC,tC,rc,cell,outc,NE0,Aw)
    print('NE cell-FE %-8s readout %.3f (boot se %.3f) z=%.2f'%(nm,ro,se,ro/se))
# insulin cell-FE: rows INS>0 with finger at t; cells = 5 mg/dL finger bins x hour block? (base constant within 5mg/dL bins)
sQ,tQ=np.nonzero(irows_pos&~np.isnan(GF))
cellI=pd.factorize(pd.Series((GF[sQ,tQ]//5).astype(int)).astype(str))[0]
ro,se,L=cellfe_readout(sQ,tQ,INS0[sQ,tQ],cellI,GF,INS0,AwI)
print('INS cell-FE finger readout %.3f (boot se %.3f) z=%.2f  L1-8 %s'%(ro,se,ro/se,L[:8].round(2)))
print('time',round(time.time()-t0,1))
```