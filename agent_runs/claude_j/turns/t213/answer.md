θ now behaves: glucose quantiles are −5.2 / 0 / +3.3 and only 0.7% of rows hit the bound. θ = 0 reproduces the counterfactual mean exactly, so tilt_mean returns a mean. The 36 mg/dL maximum error comes from the few targets outside the reachable bin range, such as x3 pushing below the lowest centre.

Next I fit the readout θ̂ = ridge(`with_code`) on glucose rows: factual rows with target 0, insulin counterfactuals, and NE counterfactuals with target "no glucose change". I use 5-fold group CV by stay and evaluate out of fold:
- the achieved mean shift per variant against the true shift;
- the factual change in NLL from the tilt.

A good outcome is out-of-fold x2/h12 achieved ≈ −12 (true −12.85), with the factual glucose NLL cost small enough that the per-reading cost averaged over all codes stays well under 0.005. A large factual cost would mean the readout can't separate counterfactual from factual histories linearly.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
EVV=M.event.values; NONE=(M.drug.values=='none')
BIN=np.array([fb_(c,x) for c,x in zip(EVV,M.obs.values)])
def pnew(ix,th,v):
    lg=torch.tensor(LG[ix])+torch.tensor(th)[:,None,None]*torch.tensor(v)[None,None,:]
    return torch.softmax(lg,-1).mean(1).numpy()
def fit_eval(msk,v,alpha,wf=1.0,nf=5):
    ix=np.where(msk)[0]; X=WC[ix]; mu=X.mean(0); sd=X.std(0)+1e-6; Z=(X-mu)/sd
    y=TH[ix]; grp=M.stay.values[ix]; w=np.where(NONE[ix],wf,1.0); oof=np.zeros(len(ix))
    for tr,te in GroupKFold(nf).split(Z,y,grp):
        oof[te]=Ridge(alpha=alpha).fit(Z[tr],y[tr],sample_weight=w[tr]).predict(Z[te])
    mn=np.asarray(tilt_mean(torch.tensor(LG[ix]),list(EVV[ix]),torch.tensor(v),torch.tensor(oof)),dtype=float).ravel()
    return ix,oof,mn
def summarize(ix,oof,mn,v,codes,drug):
    f=NONE[ix]; P0=pnew(ix[f],np.zeros(f.sum()),v); P1=pnew(ix[f],oof[f],v); b=BIN[ix[f]]; r=np.arange(f.sum())
    dn=(-np.log(P1[r,b])+np.log(P0[r,b])).mean(); nall=NONE.sum()
    o='fact n %d rms %.2f dNLL/glu %.4f dNLL/all %.4f |'%(f.sum(),np.sqrt(((mn[f]-M.mean_f.values[ix][f])**2).mean()),dn,dn*f.sum()/nall)
    for (h0,m) in ((12,0),(12,.5),(12,2),(12,3),(6,2),(16,2)):
        g=(M.drug.values[ix]==drug)&(M.h0.values[ix]==h0)&(M.m.values[ix]==m)&(EVV[ix]==codes[0])
        o+=' %g/%g S%.1f M%.1f A%.1f'%(h0,m,M.S.values[ix][g].mean(),(M.mean_cf.values[ix][g]-M.mean_f.values[ix][g]).mean(),(mn[g]-M.mean_f.values[ix][g]).mean())
    return o
for a in (10,100,1000):
    ix,oof,mn=fit_eval(isG,vG,a); print('G a',a,summarize(ix,oof,mn,vG,GLU,'insulin'))
```