The θ targets are well behaved. For insulin, 99% fall within about ±3.5. For norepinephrine they fall mostly within ±4, with about 1.5% at the bound; those are readings whose predictive distribution sits at the edge of the bins, so the requested shift cannot be reached. I will clip θ to ±5.

With gate offset G = 12, the edit unit's pre-activation stays within 7–17, well inside GELU's linear regime.

Now the decisive fit, a ridge regression of pre-activation on per-draw `with_code`. Glucose readings get G + θ: counterfactual readings their solved θ, factual readings θ = 0. MAP readings get −G, which makes the gate. λ is chosen by 5-fold CV grouped by stay. I will report:
- CV R² of θ on counterfactual samples;
- RMS of the fitted θ on factual samples (the noise that costs NLL; under about 0.25 is tolerable);
- the worst gate margins.

If R² is high and the factual noise small, the readout approach is viable.

```python
def build(drug):
    cs=GLU if drug=='insulin' else MAPC; ocs=MAPC if drug=='insulin' else GLU
    Xs,ys,gs,kind=[],[],[],[]
    for it in GEN:
        if it['drug']==drug:
            th=np.clip(it['th'],-5,5); Xs.append(it['wc'].reshape(-1,192)); ys.append(np.repeat(12+th,4)); gs.append(np.full(len(th)*4,it['s'])); kind.append(np.full(len(th)*4,0))
        elif it['drug']=='fact':
            for cset,tv,kd in [(cs,12.0,1),(ocs,-12.0,2)]:
                m=np.isin(it['ev'],cset)
                if m.sum()==0: continue
                Xs.append(it['wc'][torch.tensor(m)].reshape(-1,192)); ys.append(np.full(m.sum()*4,tv)); gs.append(np.full(m.sum()*4,it['s'])); kind.append(np.full(m.sum()*4,kd))
    return torch.cat(Xs).double().numpy(),np.concatenate(ys),np.concatenate(gs),np.concatenate(kind)
def ridge(X,y,lam):
    mx=X.mean(0); my=y.mean(); Xc=X-mx
    w=np.linalg.solve(Xc.T@Xc+lam*len(X)*np.eye(X.shape[1]),Xc.T@(y-my)); return w,my-mx@w
for drug in ['insulin','norepinephrine']:
    X,y,g,kd=build(drug); us=np.unique(g); fold=np.searchsorted(us,g)%5
    for lam in [1e-5,1e-4,1e-3,1e-2]:
        pr=np.zeros(len(y))
        for k in range(5):
            w,b=ridge(X[fold!=k],y[fold!=k],lam); pr[fold==k]=X[fold==k]@w+b
        c0=kd==0; r2=1-((pr-y)[c0]**2).mean()/((y[c0]-y[c0].mean())**2).var()**0 / ((y[c0]-y[c0].mean())**2).mean()
        print(drug[:4],lam,'cfR2',round(float(1-((pr-y)[c0]**2).mean()/((y[c0]-y[c0].mean())**2).mean()),3),
              'fact rms',round(float(np.sqrt(((pr-12)[kd==1]**2).mean())),3),'gate max',round(float(pr[kd==2].max()),1),'tgt min',round(float(pr[kd<2].min()),1))
```