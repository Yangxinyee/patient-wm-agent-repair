The fourth output of `head_run` is not the head input: its draw-average does not reproduce the stored inputs. So the per-draw check above is invalid and I will drop it. Per-draw robustness will show up directly when the edited head is evaluated.

The code gate `gvec` is clean. Glucose codes project to 1.68–1.84 and MAP codes to −0.92 to −0.76. It does not tell arterial from cuff, so for an arterial-only norepinephrine correction I need a second, code-7-specific gate.

Next I will build the gated edit from scratch:
- **Readouts.** Refit both readouts on the head input without last-bin dummies, because a gated unit cannot use the transition table. The insulin readout uses glucose rows with insulin in the preceding bundle. The norepinephrine readout uses arterial rows. Both are fitted with ridge and stay-grouped cross-validation, then scaled to unit gain.
- **Gates.** Fit a discriminant for arterial versus cuff MAP.
- **Units.** Implement each correction as a two-unit gated linear pair, u1 − u2 = r inside the gate and 0 outside.
- **Offline test.** Evaluate on the five training batches, using the same coefficients as before.

If gating works, the dNLL on rows outside the gate should be about 0. The finger-glucose slope on dev1 should stay near a_ins times the calibration, and the total cost should fall well below the ungated 0.018.

```python
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
def fit_ro(mask, ycol):
    Xr=WA_[mask].astype(np.float64); y=RA_[ycol].values[mask].astype(np.float64); grp=RA_['stay'].values[mask]
    k_=np.isfinite(y); Xr,y,grp=Xr[k_],y[k_],grp[k_]
    rc=RidgeCV(alphas=np.logspace(-3,3,13)).fit(Xr,y)
    pred=cross_val_predict(Ridge(alpha=rc.alpha_),Xr,y,cv=GroupKFold(5),groups=grp)
    gain=np.cov(pred,y)[0,1]/np.var(y,ddof=1); r2=1-np.var(y-pred)/np.var(y)
    ru=pred/gain
    print('  %s n %d alpha %.3g gain %.3f r2 %.3f | unit readout pct 0.5/50/99.5: %s'%(ycol,len(y),rc.alpha_,gain,r2,np.round(np.percentile(ru,[0.5,50,99.5]),2)))
    return dict(w=rc.coef_/gain, w0=rc.intercept_/gain, gain=gain, r2=r2)
code_=RA_['code'].values
print('readouts:')
ROi=fit_ro(np.isin(code_,[6,10])&A_.hi.values,'dev1')
ROn=fit_ro(code_==7,'nedev1')
print('gate code7 vs code5:'); g7_=lda_gate(np.isin(code_,[7,5]), code_==7)
gvn=npv(gvec).astype(np.float64)
def gated_edit(a_ins,a_ne,A=80.,K=40.,B=None,Ri=10.,Rn=10.):
    W0=ORIG['W0'].clone().double(); b0=ORIG['b0'].clone().double(); W2=ORIG['W2'].clone().double(); TR=ORIG['TR'].clone()
    u1,u2,u3,u4=units
    # glucose code gate: cg=(gv.wc-1.76)/2.6 (~0 glucose, ~-1 MAP); MAP gate cm=-(gv.wc+0.84)/2.6
    wg,cgv,sdg=gi_; Mi=(Ri+4+1.6)-0.29*A
    base_i=A*wg/sdg + K*gvn/2.6; bias_i=-A*cgv/sdg - K*1.76/2.6 + Mi
    for u,addr in [(u1,True),(u2,False)]:
        W0[u]=torch.as_tensor(base_i+(ROi['w'] if addr else 0)); b0[u]=float(bias_i+(ROi['w0'] if addr else 0))
    w7,c7,sd7=g7_; B_=B if B is not None else 80.; Mn=(Rn+4+1.6)-0.29*B_
    base_n=B_*w7/sd7 - K*gvn/2.6; bias_n=-B_*c7/sd7 - K*0.84/2.6 + Mn
    for u,addr in [(u3,True),(u4,False)]:
        W0[u]=torch.as_tensor(base_n+(ROn['w'] if addr else 0)); b0[u]=float(bias_n+(ROn['w0'] if addr else 0))
    tgt=torch.as_tensor(npv(tg),dtype=torch.float64); tmt=torch.as_tensor(npv(tm),dtype=torch.float64)
    W2[:,u1]=a_ins*tgt; W2[:,u2]=-a_ins*tgt; W2[:,u3]=a_ne*tmt; W2[:,u4]=-a_ne*tmt
    return dict(W0=W0.float(),b0=b0.float(),W2=W2.float(),TR=TR)
def eval_edit(E,label=''):
    parts=[]
    for i in range(5):
        Xs_,R_=datasets[i]; o=head_run(Xs_,R_,E); pmb,nllb,_,_=base_out[i]
        parts.append(pd.DataFrame({'sh':npv(o[0])-pmb,'dn':npv(o[1])-nllb}))
    D_=pd.concat(parts,ignore_index=True); D_['code']=A_.code.values; D_['drug']=A_.drug.values; D_['dev1']=A_.dev1.values; D_['nedev1']=A_.nedev1.values
    def slope(g,xc):
        x=g[xc].values; y=g.sh.values; k_=np.isfinite(x); return np.cov(x[k_],y[k_])[0,1]/np.var(x[k_],ddof=1)
    tab_=D_.groupby(['code','drug']).apply(lambda g:pd.Series({'n':len(g),'dNLL':g.dn.mean(),'sh_mean':g.sh.mean(),'sh_sd':g.sh.std(),
            'slope':slope(g,'dev1' if g.code.iloc[0] in (6,10) else 'nedev1')}))
    print(label,'overall dNLL %.4f'%D_.dn.mean()); print(tab_.round(4)); return D_
E0g=gated_edit(0.0,0.0); _=eval_edit(E0g,'a=0 (units removed):')
Eg=gated_edit(-1.9,3.0); Dg=eval_edit(Eg,'gated a_ins=-1.9 a_ne=3:')
```