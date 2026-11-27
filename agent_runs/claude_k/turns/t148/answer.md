The unbiased decoder fails badly: its cross-validated R² is −7 to −48. So q's response to a dose change, with readings held fixed, is nearly collinear with its response to the readings themselves. The model files the dose almost in the same place as "glucose level".

Any linear readout of q that tracks the dose itself is therefore very noisy. A shrunk ridge probe moves only about 0.18 per unit of insulin and 0.11 per unit of norepinephrine. To give the right response it would need roughly 5–9× gain, which would also amplify its noise and its response to readings.

Before choosing between this value-head route and a harder upstream edit (giving the state a clean dose channel), I'll measure the actual trade-off on the 99 factual stays. I'll use out-of-fold ridge probes for the lag-1 and lag-2 residuals, combined with the factual β weights. Then I'll compute the exact value-NLL change for gains λ = 1–8. The shift is applied as an exponential tilt over bin centres, as an edit would apply it, and λ × probe response gives the implied counterfactual sensitivity.

- If λ ≈ 5 costs well under 0.002 nats per reading for each drug, the value-head route is viable and I'll build it.
- If it costs more, the budget allows only a partial gain, and I'll weigh the upstream channel.

```python
def oof_ridge(Xm,y,grp,al=10.0):
    pr=np.zeros(len(y))
    for tri,tei in GroupKFold(5).split(Xm,y,grp):
        m=Ridge(alpha=al).fit(Xm[tri],y[tri]); pr[tei]=m.predict(Xm[tei])
    return pr, Ridge(alpha=al).fit(Xm,y)
cenfull=np.column_stack([np.zeros(len(CENm)),CENm])
def tilt_nll(delta):  # shift predicted mean of each reading by delta via one-step tilt on centres
    c=cenfull; p=np.exp(LPA); m=(p*c).sum(1); v=(p*(c-m[:,None])**2).sum(1)
    eta=delta/np.maximum(v,1e-6); lg2=LPA+eta[:,None]*(c-m[:,None]); L2=lsm_(lg2)
    return -L2[np.arange(len(obA)),obA]
grp_all=M4.stay_id.values
res_tab={}
for nm_,msk_,R_,bet in [('INS',glu,RI_,(-1.72,-1.66)),('NE',mapm,RN_,(2.68,2.69))]:
    s=np.zeros(len(obA)); resp=0
    Bq,*_=np.linalg.lstsq(np.column_stack([np.ones(msk_.sum()),R_[msk_][:,:3],zl[msk_]]),WCA[msk_],rcond=None)
    for l in range(2):
        pr,m=oof_ridge(WCA[msk_],R_[msk_][:,l],grp_all[msk_]); s[msk_]+=bet[l]*pr; resp+=bet[l]*(m.coef_@Bq[1+l])
    base=nllA.mean()
    out=[(lam, round(tilt_nll(lam*s).mean()-base,5)) for lam in [1,2,3,5,8]]
    print(nm_,'sum beta',sum(bet),' probe resp per unit sustained dose (lam=1): %.3f'%resp, ' dNLL(all readings) by lam', out)
```