Fitting factual residuals did not work. The decoder learned mostly noise: the factual tilt has an sd of about 4 mg/dL out of sample, glucose NLL gets worse by 0.02–0.05 nats, and the counterfactual slopes stay at 0.1–0.6. With 7,000 factual rows and noisy residuals, 193 features overfit.

Factual gains are not what is scored anyway. The edit is judged on the counterfactual readouts, with factual fit only a constraint. So I'll try the opposite design: fit the decoder only to the counterfactual arm changes, and penalise its spread on factual rows. The intercept is set so the mean factual tilt is zero.

I estimate the glucose predictive sd at about 12.6 mg/dL (from NLL 1.47 at the bin widths). That means a factual tilt noise of about 1.5 mg/dL costs roughly 0.007 nats per glucose reading. This is about the most I can afford, since glucose rows are only part of the 0.005-per-reading budget.

For a range of variance penalties μ I'll report the factual NLL change, the factual tilt sd, and the per-arm slopes. A μ with slopes of about 0.8–1 for x2, x1/2 and x1.5 and a glucose NLL change of at most about 0.005 would make the head-only edit feasible. Otherwise I'll look into the upstream channel via the prelude attention.

```python
Xfc=Xw[f0]-Xw[f0].mean(0)
for mu in [0.0,0.3,3.0,30.0]:
    rf=np.zeros(len(f0)); rc=np.zeros(len(pairs))
    for fo in [0,1]:
        trf=fold[f0]!=fo; trc=fold[pairs[:,0]]!=fo
        Xc_=Xc[trc][:,:192]
        A_=Xc_.T@Xc_+mu*Xfc[trf].T@Xfc[trf]+1e-3*np.eye(192); b_=Xc_.T@tc[trc]
        w_=np.linalg.solve(A_,b_)
        r_all=Xw[f0]@w_; r_all=r_all-r_all[trf].mean()
        rf[~trf]=r_all[~trf]; rc[~trc]=Xc[~trc][:,:192]@w_
    Pn=tilt_P(PD[f0],cenD[f0],rf); dn=nll_of(Pn,obD[f0]).mean()-nll0
    sl=[round(float((rc[pairs[:,2]==a]*tc[pairs[:,2]==a]).sum()/(tc[pairs[:,2]==a]**2).sum()),2) for a in range(1,6)]
    print('mu',mu,'dNLL glucose %.4f'%dn,'tilt sd fact %.2f'%rf.std(),'corr(rf, true dev-signal) %.2f'%np.corrcoef(rf,(devD[f0]@G85))[0,1],'cf slopes',sl)
```