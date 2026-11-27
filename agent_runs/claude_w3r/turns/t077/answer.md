Type 1 is now best for both glucose (2.1 mg/dL) and MAP (0.94 mmHg), so the type question is settled: row 1 is the measurement query. Averaging only 4 z draws by itself scatters predictions by about 1.0 mg/dL for glucose and 0.5 mmHg for MAP. My residual is about twice that floor, so a small structural detail is still off.

The remaining candidates are:
- whether the head's gap term uses log(dt)/10 or plain log(dt);
- the age unit, and whether age is measured to the predicted bundle or to the last bundle.

I test these combinations on more readings. If one brings both errors down to the noise floor (about 1.0 and 0.5), the head is exact. If none does, I will accept the head as close enough and look elsewhere.

```python
def prep2(HI_, ns=32, gapdiv=10.0, seed=0):
    P=[]; g_=torch.Generator().manual_seed(seed)
    for (i0,dt,lv,age) in HI_:
        out_=torch.tensor(S_['x'][i0,:192]); h_=torch.tensor(S_['x'][i0,192:960]).reshape(4,192)
        hf=h_*torch.exp(-dt*torch.exp(fa)); ctxz=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
        pr=MLP(ctxz,'z_prior','gelu'); zs=pr[:32]+torch.exp(pr[32:])*torch.randn(ns,32,generator=g_)
        viewb=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/gapdiv],dtype=torch.float32),'gap','gelu')
        P.append((viewb+L_(zs,'z_head'), lv, age, dt))
    return P
HIg=HI[:150]; obs_g=fs.predicted_mean.values[:150]
for gapdiv in [10.0, 1.0]:
    Pg=prep2(HIg, gapdiv=gapdiv)
    for ad in [60.0, 3600.0, 600.0]:
        for agemode in ['to_pred','to_last']:
            P2=[(v,lv,(age if agemode=='to_pred' or age is None else max(age-dt,0.0)),) for (v,lv,age,dt) in Pg]
            e_=np.median(np.abs(pm_from(P2,'glucose_finger',1,(0,1,2,3),0,ad)-obs_g))
            print('gapdiv',gapdiv,'age_div',ad,agemode,'glucose med err',round(float(e_),2))
```