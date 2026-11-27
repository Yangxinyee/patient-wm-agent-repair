Dropping the gap term (version e, error 0.14) or the time flow (version b, 0.17) both beat my current version (0.22). So the units or scaling of the time gap are probably wrong somewhere.

Next I'll scan the time-flow units (seconds, minutes, hours, none) crossed with the gap-encoder input (several unit and scaling choices, or none). A combination giving near-zero error would fix both conventions.

```python
res_=[]
gopts={'none':None,'ls10':lambda d:np.log(d)/10,'lm10':lambda d:np.log(d/60)/10,'lh10':lambda d:np.log(d/3600)/10,'ls':lambda d:np.log(d),'lh':lambda d:np.log(d/3600),'l1ph10':lambda d:np.log1p(d/3600)/10}
for s1 in [0,1/3600,1/60,1.0]:
    for gk,gf in gopts.items():
        E_=[]
        for p_,c_,dt,e_ in cases:
            out_=X0[p_,:192]; hf=X0[p_,192:960].reshape(4,192)*torch.exp(-s1*dt*flow)
            ctx=out_+lnp(hf.mean(0),'core.out_norm')+(gapx(gf(dt)) if gf else 0)
            E_.append((mlp(torch.cat([ctx,e_,torch.zeros(192)]),'z_post')[:32]-X0[c_,960:992]).numpy())
        res_.append((round(float(np.sqrt((np.array(E_)**2).mean())),4),s1,gk))
res_.sort(); print(res_[:6])
```