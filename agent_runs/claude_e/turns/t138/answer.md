That attempt was too blunt. Adding zero targets on every non-target row, including glucose rows with no insulin, collapses the insulin readout from R² 0.60 to 0.18. Those rows sit right next to the target rows in input space.

The all-negatives discriminant also fails for insulin: the minority boundary between glucose rows with and without insulin gets swamped by MAP rows. The arterial gate against all other codes is clean (±1 standardised margin), so I keep that one.

The middle ground: add zero-target rows only from other codes, where the code embedding separates them cleanly. The insulin readout gets MAP rows. The norepinephrine readout gets glucose and cuff rows. Glucose rows without insulin are left out of the fit and are handled by the within-glucose insulin gate together with the code gate.

Next I will:
1. Refit both readouts this way on per-draw inputs, choosing alpha by stay-grouped CV.
2. Refit the within-glucose insulin gate on per-draw inputs.
3. Report readout ranges in every row group.
4. Pick gate scales and offsets numerically, so the on rows are linear and the off rows are dead, with pre-activations kept moderate.
5. Check the pair outputs directly before running the head.

I expect the insulin readout's R² to stay near 0.6, the off-code readout to be bounded to a few units, and every gated-off group to give about 0 output on all draws.

```python
def fit_ro3(tmask,zmask,ycol,alphas=(1e-3,1e-2,1e-1,1.0),ndr=4):
    idx=np.where(tmask|zmask)[0]; y_full=np.where(tmask,np.nan_to_num(A_[ycol].values),0.0)
    Xs=np.concatenate([WCall[d][idx] for d in range(ndr)],0).astype(np.float64)
    ys=np.tile(y_full[idx],ndr); gs=np.tile(stayA[idx],ndr); ts=np.tile(tmask[idx],ndr)
    best=None
    for al in alphas:
        pr=np.zeros(len(ys))
        for trn,tst in GroupKFold(5).split(Xs,ys,gs): pr[tst]=Ridge(alpha=al).fit(Xs[trn],ys[trn]).predict(Xs[tst])
        r2=1-np.var(ys[ts]-pr[ts])/np.var(ys[ts]); gain=np.cov(pr[ts],ys[ts])[0,1]/np.var(ys[ts],ddof=1)
        if best is None or r2>best[1]: best=(al,r2,gain)
    al,r2,gain=best; rid=Ridge(alpha=al).fit(Xs,ys)
    w=rid.coef_/gain; w0=rid.intercept_/gain
    R=np.einsum('dnk,k->dn',WCall,w.astype(np.float32))+w0
    print('  %s alpha %g r2 %.3f gain %.3f'%(ycol,al,r2,gain))
    return dict(w=w,w0=w0,gain=gain,r2=r2),R
def lda_in(pmask,nmask,lam=1e-2,ndr=4):
    idx=np.where(pmask|nmask)[0]
    Xs=np.concatenate([WCall[d][idx] for d in range(ndr)],0).astype(np.float64); ys=np.tile(pmask[idx],ndr)
    mu1=Xs[ys].mean(0); mu0=Xs[~ys].mean(0); S=np.cov(Xs.T)+lam*np.eye(192); w=np.linalg.solve(S,mu1-mu0)
    P=np.einsum('dnk,k->dn',WCall,w.astype(np.float32))
    lo=np.percentile(P[:,pmask],0.05); hi_=np.percentile(P[:,nmask],99.95); c=(lo+hi_)/2; s=(lo-hi_)/2
    return dict(w=w/s,c=c/s,sep=s),(P-c)/s
glu=np.isin(codeA,[6,10]); mapc=np.isin(codeA,[7,5])
print('insulin readout (zeros on MAP rows):'); ROi3,Ri3=fit_ro3(ins_t,mapc,'dev1')
print('NE readout (zeros on glucose + cuff rows):'); ROn3,Rn3=fit_ro3(ne_t,~ne_t,'nedev1')
Gi3,Pi3=lda_in(ins_t,glu&~hiA)
print('insulin gate within glucose: sep %.3f'%Gi3['sep'])
grp={'glu&hi':ins_t,'glu&~hi':glu&~hiA,'MAP':mapc,'code7':ne_t,'nonc7':~ne_t}
for nm,R_,gs_ in [('Ri3',Ri3,['glu&hi','glu&~hi','MAP']),('Rn3',Rn3,['code7','nonc7'])]:
    print(nm,{g:np.round(np.percentile(R_[:,grp[g]],[0.05,0.5,99.5,99.95]),2).tolist() for g in gs_})
print('Pi3 (ins gate) pct:',{g:np.round(np.percentile(Pi3[:,grp[g]],[0.05,50,99.95]),2).tolist() for g in ['glu&hi','glu&~hi','MAP']})
cgA=(np.einsum('dnk,k->dn',WCall,gvn.astype(np.float32))-1.76)/2.6
print('code gate pct:',{g:np.round(np.percentile(cgA[:,grp[g]],[0.05,50,99.95]),3).tolist() for g in ['glu&hi','glu&~hi','MAP']})
print('NE gate Pn_ pct:',{g:np.round(np.percentile(Pn_[:,grp[g]],[0.05,50,99.95]),2).tolist() for g in ['code7','nonc7']})
```