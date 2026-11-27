Restricting the zero targets to other codes still collapses the insulin readout (R² 0.22).

The reason is structural. The head input is a sum: a normalised state term that is the same for every code at a given time, plus code, last-bin and last-value terms. The dose information sits in the shared state term. So any readout forced to be zero on MAP rows must also ignore the dose on glucose rows.

The fix is to leave the readouts' fitting rows alone and remove only the code part of the input. I will fit each readout on its target rows alone, on per-draw inputs, with weights orthogonal to the differences between code embeddings. That removes the large constant offset (about +160) the insulin readout had on MAP rows. It should cost almost nothing on the target rows, where the code is constant.

Gating is then left to the discriminants:
- a within-glucose insulin gate with large separation (medians +5.7 and −11.9 on the standardised scale);
- a cleaner glucose-versus-MAP code gate, which I will check for spread;
- the arterial gate.

I expect the projected readouts to keep R² near 0.6 for insulin and 0.18 for norepinephrine, and to stay bounded on all off-target groups.

```python
EC=np.asarray(load_original('embed.code.weight'),dtype=np.float64); print('embed.code',EC.shape)
def perp(ref,others):
    Dm=np.stack([EC[o]-EC[ref] for o in others],1); Qd,_=np.linalg.qr(Dm); return np.eye(192)-Qd@Qd.T
def fit_ro4(tmask,ycol,Pp,alphas=(1e-3,1e-2,1e-1),ndr=4):
    idx=np.where(tmask)[0]; y=np.nan_to_num(A_[ycol].values[idx])
    Xs=np.concatenate([WCall[d][idx] for d in range(ndr)],0).astype(np.float64)@Pp
    ys=np.tile(y,ndr); gs=np.tile(stayA[idx],ndr); best=None
    for al in alphas:
        pr=np.zeros(len(ys))
        for trn,tst in GroupKFold(5).split(Xs,ys,gs): pr[tst]=Ridge(alpha=al).fit(Xs[trn],ys[trn]).predict(Xs[tst])
        r2=1-np.var(ys-pr)/np.var(ys); gain=np.cov(pr,ys)[0,1]/np.var(ys,ddof=1)
        if best is None or r2>best[1]: best=(al,r2,gain)
    al,r2,gain=best; rid=Ridge(alpha=al).fit(Xs,ys)
    w=(Pp@rid.coef_)/gain; w0=rid.intercept_/gain
    R=np.einsum('dnk,k->dn',WCall,w.astype(np.float32))+w0
    print('  %s alpha %g r2 %.3f gain %.3f'%(ycol,al,r2,gain)); return dict(w=w,w0=w0,gain=gain,r2=r2),R
print('insulin readout, code-invariant:'); ROi4,Ri4=fit_ro4(ins_t,'dev1',perp(6,[10,7,5]))
print('NE readout, code-invariant:'); ROn4,Rn4=fit_ro4(ne_t,'nedev1',perp(7,[5,6,10]))
for nm,R_,gs_ in [('Ri4',Ri4,['glu&hi','glu&~hi','MAP']),('Rn4',Rn4,['code7','nonc7'])]:
    print(nm,{g:np.round(np.percentile(R_[:,grp[g]],[0.05,0.5,99.5,99.95]),2).tolist() for g in gs_}, 'max|off|',round(float(np.abs(R_[:,~grp[gs_[0]]]).max()),2))
Gc,Pc=lda_in(glu,mapc)
med_g=np.median(Pc[:,glu]); gcz=(Pc-med_g)/(med_g-np.percentile(Pc[:,mapc],99.95))
print('code gate LDA: glucose spread pct0.05/99.95 %s | MAP max %.3f'%(np.round(np.percentile(gcz[:,glu],[0.05,99.95]),4),gcz[:,mapc].max()))
```