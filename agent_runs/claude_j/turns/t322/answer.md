Ridge regression on the dose deviation is the wrong tool. The drug-I probe's own noise (about ±0.35 on bundles without the drug) is comparable to the true deviation, so ridge puts little weight on the channel. That is why the x2 slope came out at only 0.15–0.19.

The channel itself is not tiny. Its response is about 0.09 (drug I) and 0.32 (drug N) in norm per unit rate, and GPU precision noise along the readouts is only 0.04–0.07.

The right construction separates two jobs:
- **(a) A Fisher readout g of the channel's mean response direction J**, normalised to g·J = 1 so it has slope 1 on dose changes. Its noise metric includes the measured GPU–replica precision covariance.
- **(b) A centering term β that predicts the context-expected dose from the original head input**, constrained to β·J = 0 so it cannot cancel the dose response.

The unit then reads (g − β)·WC ≈ dose deviation.

This run checks:
- the x2 slope, which should be ≈ 1;
- the noise along g;
- the correlation of the centered unit with the true deviation;
- the factual-optimal gain from residuals, which should be negative for drug I on glucose and positive for drug N on MAP.

I'll also print `HT` from heads.py to see whether the replica heads read saved edits.

```python
import re
src=open('/work/heads.py').read(); print(re.search(r'def HT\(n\):.*?\n(.*?)\n',src,re.S).group(0)[:200])
CP=np.cov(GP.T); FR={}
for dr,(cods,dc,rc) in GR.items():
    m_=RW['ev'].isin(cods).values; rr=RW[rc].values[m_]; po=RW['post'].values[m_]; imm_=rr>0
    W0=np.concatenate([D3[s]['wc0'] for s in stays])[m_]; W1=WC1[m_]; Wc=WCc[dr][m_]; dW=W1-W0
    J_=(dW[imm_]/rr[imm_,None]).mean(0); Sg=np.cov(W0.T)+CP+1e-6*np.eye(192); g=np.linalg.solve(Sg,J_); g/=g@J_
    mu_=W0.mean(0); sl=np.polyfit(rr[po],(Wc-W1)[po]@g,1)[0]; nsd=float(((W0-mu_)@g).std())
    Pj=np.eye(192)-np.outer(J_,J_)/(J_@J_); Ehat=np.concatenate([np.zeros(0)]+[])
    ex=RW[dc].values[m_]; ehat=rr-ex   # Ehat at reading (=r-dev) for rows with recent drug; 0 otherwise
    sel=rr>0; Z=(W0-mu_)@Pj; bm=Ridge(alpha=1.0).fit(Z[sel],ehat[sel]); bcv=cross_val_predict(Ridge(alpha=1.0),Z[sel],ehat[sel],cv=5)
    beta=Pj@bm.coef_; u=(W1-mu_)@(g-beta)-bm.intercept_*(rr>0)
    c=np.corrcoef(u[sel],ex[sel])[0,1]; res=OB[m_]-PM1[m_]
    print(dr[:3],'x2slope',round(float(sl),3),'noise sd',round(nsd,3),'prec sd',round(float((GP[np.isin(evs40,cods)]@g).std()),3),'Ehat cvR2',round(1-((bcv-ehat[sel])**2).sum()/((ehat[sel]-ehat[sel].mean())**2).sum(),3),'corr(u,dev)',round(float(c),3))
    for c_ in cods:
        mc=RW['ev'].values[m_]==c_; print('   ',c_[-6:],'A',round(float(np.polyfit(u[mc],res[mc],1)[0]),2),'sd u',round(float(u[mc].std()),3))
    FR[dr]=dict(g=g,beta=beta,b0=float(bm.intercept_),mu=mu_,J=J_)
```