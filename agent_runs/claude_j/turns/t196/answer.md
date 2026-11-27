Most of the factual "noise" is constant offsets, not noise. Even at λ = 0.01, where the readout barely uses the signal (counterfactual R² 0.07), the factual RMS is 0.32. That points to code- and last-value-bin offsets from the gate: glucose_finger and glucose_lab sit at different gate levels. Those constants will be cancelled exactly in each code's `embed_transition` rows, so the real cost is the spread within each (code, last bin) group.

The readout recovers a fair share of the counterfactual signal (R² 0.52 for insulin, 0.39 for norepinephrine at λ = 1e-5). The rollout readout averages shifts, so calibration of the means matters more than per-reading R².

For each λ, from out-of-fold predictions, I will report:
- the factual RMS within (code, last bin) groups, after removing group means estimated on factual samples only;
- the mean fitted θ against the target mean for each counterfactual condition (drug, factor, start hour);
- the gate margin.

A within-group RMS under about 0.2 and condition means tracking their targets would make the approach viable.

```python
def diag(drug,lam):
    X,y,g,kd=build(drug); us=np.unique(g); fold=np.searchsorted(us,g)%5; pr=np.zeros(len(y))
    for k in range(5):
        w,b=ridge(X[fold!=k],y[fold!=k],lam); pr[fold==k]=X[fold==k]@w+b
    # group ids for target samples: code embedding + last bin identified via rounding of features is messy -> rebuild group keys
    keys=[]
    for it in GEN:
        if it['drug']==drug or it['drug']=='fact':
            cs=GLU if drug=='insulin' else MAPC; ocs=MAPC if drug=='insulin' else GLU
            sets=[None] if it['drug']==drug else [cs,ocs]
            for cset in sets:
                m=np.ones(len(it['ev']),bool) if cset is None else np.isin(it['ev'],cset)
                if m.sum()==0: continue
                rt=RTs[it['s']].iloc[it['idx'][m]] if it['drug']==drug else RTs[it['s']][m]
                _,lb,cid=last_feats(rt); keys.append(np.repeat(cid.numpy()*17+lb.numpy(),4))
    keys=np.concatenate(keys)
    fm=kd==1; gm=pd.Series(pr[fm]-12).groupby(keys[fm]).mean()
    adj=pr-12-pd.Series(keys).map(gm).fillna(0).values
    within=float(np.sqrt((adj[fm]**2).mean()))
    th_hat=adj[kd==0]; th=y[kd==0]-12
    return within, th_hat, th, float(pr[kd==2].max())
for drug in ['insulin','norepinephrine']:
    for lam in [1e-5,1e-4,1e-3]:
        wi,thh,th,gmax=diag(drug,lam)
        print(drug[:4],lam,'fact within-rms',round(wi,3),'cf mean th_hat/th',round(float(thh.mean()),3),round(float(th.mean()),3),'slope',round(float(np.polyfit(th,thh,1)[0]),2),'gate',round(gmax,1))
```