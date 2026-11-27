Mystery solved. The head's hidden output equals GELU of my computed pre-activation (error 6e-12), so the local head arithmetic is right. On arterial MAP rows, the gate terms add up to about −68, yet the pre-activation is +91.

The difference is the readout itself. I fitted it only on glucose rows after an insulin bundle, with almost no regularisation. Off that domain, on MAP rows, it reaches about +160, enough to override the gates. The lab-glucose leakage of the norepinephrine pair is the same problem.

The fix is to bound each readout where it should be silent. I will refit each readout on per-draw inputs from all rows: the target rows keep their true deviation and all other rows get target 0. Each gate will be fitted against all other rows, not only the same code group.

I will keep pre-activations moderate, within roughly ±100, so the arithmetic stays safe in float32. I will then report:
- readout R² and gain on the target rows;
- readout range on the other rows;
- the per-draw gate margins.

The aim is readouts bounded to a few units off-target and clean separation on all 16 draws.

```python
parts=[]
for i in range(5):
    parts.append(wc_draws(i).astype(np.float32))
WCall=np.concatenate(parts,axis=1); del parts
print('WCall',WCall.shape)
codeA=A_.code.values; hiA=A_.hi.values; stayA=A_.stay.values
def fit_ro2(tmask,ycol,wneg=1.0,alpha=1e-2,ndr=4):
    y_full=np.where(tmask,np.nan_to_num(A_[ycol].values),0.0)
    Xs=np.concatenate([WCall[d] for d in range(ndr)],0).astype(np.float64); ys=np.tile(y_full,ndr); ws=np.tile(np.where(tmask,1.0,wneg),ndr); gs=np.tile(stayA,ndr)
    rid=Ridge(alpha=alpha).fit(Xs,ys,sample_weight=ws)
    pred=cross_val_predict(Ridge(alpha=alpha),Xs,ys,cv=GroupKFold(5),groups=gs,params={'sample_weight':ws}) if False else None
    # CV on target rows only via group folds over stays
    gkf=GroupKFold(5); pr=np.zeros(len(ys))
    for trn,tst in gkf.split(Xs,ys,gs):
        pr[tst]=Ridge(alpha=alpha).fit(Xs[trn],ys[trn],sample_weight=ws[trn]).predict(Xs[tst])
    tm_=np.tile(tmask,ndr)
    gain=np.cov(pr[tm_],ys[tm_])[0,1]/np.var(ys[tm_],ddof=1); r2=1-np.var(ys[tm_]-pr[tm_])/np.var(ys[tm_])
    w=rid.coef_/gain; w0=rid.intercept_/gain
    R=np.einsum('dnk,k->dn',WCall,w.astype(np.float32))+w0
    print('  %s: gain %.3f r2 %.3f | target pct0.5/99.5 %s | off-target pct0.1/99.9 %s max|.| %.1f'%(ycol,gain,r2,np.round(np.percentile(R[:,tmask],[0.5,99.5]),2),np.round(np.percentile(R[:,~tmask],[0.1,99.9]),2),np.abs(R[:,~tmask]).max()))
    return dict(w=w,w0=w0,gain=gain,r2=r2),R
def lda2(tmask,lam=1e-2,ndr=4):
    Xs=np.concatenate([WCall[d] for d in range(ndr)],0).astype(np.float64); ys=np.tile(tmask,ndr)
    mu1=Xs[ys].mean(0); mu0=Xs[~ys].mean(0); S=np.cov(Xs.T)+lam*np.eye(192); w=np.linalg.solve(S,mu1-mu0)
    P=np.einsum('dnk,k->dn',WCall,w.astype(np.float32))
    lo=np.percentile(P[:,tmask],0.05); hi_=np.percentile(P[:,~tmask],99.95); c=(lo+hi_)/2; s=(lo-hi_)/2
    P=(P-c)/s
    print('  gate: target min %.2f pct0.05 %.2f | off pct99.95 %.2f max %.2f | per-draw sd %.3f'%(P[:,tmask].min(),np.percentile(P[:,tmask],0.05),np.percentile(P[:,~tmask],99.95),P[:,~tmask].max(),P.std(0).mean()))
    return dict(w=w/s,c=c/s),P
ins_t=np.isin(codeA,[6,10])&hiA; ne_t=codeA==7
print('insulin:'); ROi2,Ri_=fit_ro2(ins_t,'dev1'); Gi2,Pi_=lda2(ins_t)
print('NE:'); ROn2,Rn_=fit_ro2(ne_t,'nedev1'); Gn2,Pn_=lda2(ne_t)
```