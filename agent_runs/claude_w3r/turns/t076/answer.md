The best configuration is has_last, log(1+age in minutes)/10, z_last_value and a has_value flag in the first four inputs, with last_status 0. It gives a median error of 2.1 mg/dL for glucose and 0.94 mmHg for MAP, the best so far. Some of what remains may simply be the model's own noise: each of its predictions averages only 4 random z draws, while mine averages 32.

Next I measure how far 4-draw predictions scatter around my 32-draw prediction, using this configuration and several seeds. I also re-scan the type index under it. If the 4-draw scatter is about 2 mg/dL and 1 mmHg, my head is essentially exact. If it is much smaller, there is still a structural gap.

```python
BEST=dict(layout=(0,1,2,3), lstat=0, ad=60.0)
def prep_s(HI_, ns, seed):
    P=[]; g_=torch.Generator().manual_seed(seed)
    for (i0,dt,lv,age) in HI_:
        out_=torch.tensor(S_['x'][i0,:192]); h_=torch.tensor(S_['x'][i0,192:960]).reshape(4,192)
        hf=h_*torch.exp(-dt*torch.exp(fa)); ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
        pr=MLP(ctx,'z_prior','gelu'); zs=pr[:32]+torch.exp(pr[32:])*torch.randn(ns,32,generator=g_)
        P.append((ctx+L_(zs,'z_head'), lv, age))
    return P
base_g=pm_from(PG,'glucose_finger',1,BEST['layout'],0,60.0)
sp=[np.median(np.abs(pm_from(prep_s(HI[:60],4,s),'glucose_finger',1,BEST['layout'],0,60.0)-base_g)) for s in [1,2,3]]
print('glucose 4-draw vs 32-draw median abs dev', np.round(sp,2))
base_m=pm_from(PM,'map_arterial',1,BEST['layout'],0,60.0)
spm=[np.median(np.abs(pm_from(prep_s(HM,4,s),'map_arterial',1,BEST['layout'],0,60.0)-base_m)) for s in [1,2]]
print('map 4-draw dev', np.round(spm,2))
print('types glucose', [round(float(np.median(np.abs(pm_from(PG,'glucose_finger',t,BEST['layout'],0,60.0)-obs_pm[:60]))),2) for t in range(15)])
print('types map', [round(float(np.median(np.abs(pm_from(PM,'map_arterial',t,BEST['layout'],0,60.0)-fm.predicted_mean.values))),2) for t in range(15)])
```