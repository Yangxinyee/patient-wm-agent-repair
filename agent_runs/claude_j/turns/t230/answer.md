Even on exposed rows after hour 12, the linear readout costs about 0.06–0.07 nats per glucose reading, roughly six times the whole budget. So exposure gating alone won't rescue it.

That leaves two possibilities:
1. `with_code` does carry the counterfactual signal, but not linearly, so a nonlinear readout could separate counterfactual from factual states.
2. The signal isn't in the head input at all.

Two diagnostics in one run:
- **A.** Linear ridge with honest factual targets: the observed residual, linearized through the tilt gain. This tests whether the readout's factual output can be right rather than forced to 0. If the factual NLL change falls to about 0 or below, the readout captures real deviation structure.
- **B.** A gradient-boosting regressor on `with_code`. If it shows much lower factual cost at similar recovery, the information exists nonlinearly and a multi-unit readout is worth building. If it is just as bad, a head-only edit can't work and I must move upstream.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
ixG=np.where(isG)[0]; ev=list(EVV[ixG]); LGt=torch.tensor(LG[ixG]); vt=torch.tensor(vG)
gp=np.asarray(tilt_mean(LGt,ev,vt,torch.full((len(ixG),),0.05,dtype=torch.float64)),float); gm=np.asarray(tilt_mean(LGt,ev,vt,torch.full((len(ixG),),-0.05,dtype=torch.float64)),float)
GAIN=np.zeros(len(M)); GAIN[ixG]=(gp-gm)/0.1
print('gain q10/50/90',np.round(np.quantile(GAIN[ixG],[.1,.5,.9]),2))
def oof_fit(msk,y,w,model,nf=5):
    ix=np.where(msk)[0]; X=WC[ix]; Z=(X-X.mean(0))/(X.std(0)+1e-6); oof=np.zeros(len(ix))
    for tr,te in GroupKFold(nf).split(Z,y[ix],M.stay.values[ix]):
        mdl=model(); mdl.fit(Z[tr],y[ix][tr],sample_weight=w[ix][tr]); oof[te]=mdl.predict(Z[te])
    return ix,oof
yA=TH.copy(); fr=NONE&isG; yA[fr]=np.clip((M.obs.values[fr]-M.mean_f.values[fr])/np.maximum(GAIN[fr],1e-3),-8,8)
wA=np.maximum(GAIN,1e-3)**2
for nm,y,w,mdl in (('A ridge obs-targets',yA,wA,lambda:Ridge(alpha=100)),('B hgb',TH,np.where(NONE,5.0,1.0),lambda:HistGradientBoostingRegressor(max_iter=200,learning_rate=0.1))):
    t0=time.time(); ix,oof=oof_fit(isG,y,w,mdl); mn=np.asarray(tilt_mean(torch.tensor(LG[ix]),list(EVV[ix]),vt,torch.tensor(oof)),float)
    f=NONE[ix]; h=M.hour.values[ix]; st=M.stay.values[ix]; ex=np.array([h_>first[s_] for h_,s_ in zip(h,st)])
    print(nm,'%.0fs'%(time.time()-t0),dnll_groups(ix,oof,vG,(('exp',f&ex),('unexp',f&~ex))),'|',summarize(ix,oof,mn,vG,GLU,'insulin').split('|')[1])
```