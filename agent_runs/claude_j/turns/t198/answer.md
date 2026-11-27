The native signal cannot be read cleanly. The counterfactual change in the head query is almost rank 2, but recorded data vary 15–30 times more along exactly those directions (signal-to-noise 0.04–0.07). The model stores dose information where it is swamped by patient-state variation.

My factual-noise estimate ignored one thing. Factual dose deviations are real exogenous shifts with real effects, so part of the readout's factual output may be correct correction and improve NLL.

Before committing to a much harder state-level injection, I will measure the actual factual NLL change. I will apply the out-of-fold fitted units, after group-mean cancellation, at several gains to the recorded readings. If even full gain costs less than about 0.005 nats per reading, the head-level approach is acceptable after all. Otherwise I will learn how much gain the budget allows.

```python
def oof_units(drug,lam=1e-5):
    X,y,g,kd=build(drug); us=np.unique(g); fold=np.searchsorted(us,g)%5; W_={}
    for k in range(5): W_[k]=ridge(X[fold!=k],y[fold!=k],lam)
    return W_,us
UN={d_:oof_units(d_) for d_ in ['insulin','norepinephrine']}
def fact_nll(gains):
    tot=0.0; base=0.0; n=0
    for it in GEN:
        if it['drug']!='fact': continue
        s=it['s']; rt=RTs[s]; obs=np.array([fb_(c,v) for c,v in zip(rt.event,rt.observed)])
        lg=it['lg'].double(); add=torch.zeros_like(lg)
        for d_,v_ in [('insulin',VIN),('norepinephrine',VNE)]:
            W_,us=UN[d_]; k=int(np.searchsorted(us,s)%5); w,b=W_[k]
            m=np.isin(rt.event.values,GLU if d_=='insulin' else MAPC)
            pre=it['wc'].double().numpy()@w+b; th=torch.tensor(pre-12)*torch.tensor(m)[:,None]
            add+=gains[d_]*th[:,:,None]*v_
        it['_th']=add
        P0=torch.softmax(lg,-1).mean(1); P1=torch.softmax(lg+add,-1).mean(1)
        idx=torch.arange(len(obs)); base+=float(-torch.log(P0[idx,obs]).sum()); tot+=float(-torch.log(P1[idx,obs]).sum()); n+=len(obs)
    return (tot-base)/n
for gI,gN in [(1,0),(0,1),(1,1),(2,2)]:
    print('gains',gI,gN,'dNLL/reading (no group-mean cancel)',round(fact_nll({'insulin':gI,'norepinephrine':gN}),4))
```