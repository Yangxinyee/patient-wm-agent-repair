The best combination puts has_last, log age and z_last_value in the first three inputs and leaves the rest at zero, with last_status 0 and type row 10. It is off by a median of 2.5 mg/dL, so closer but not exact. Part of the remaining gap may come from z: the model averages probabilities over 4 random prior draws of z, while I used the prior mean.

Next, for that layout, I average the probabilities over 32 prior samples, scan all 15 types again, and run the same check on map_arterial.
- If the error then falls to roughly the size of the model's own 4-draw noise, and the best type is the same for glucose and MAP, the head is validated.
- If glucose and MAP prefer different types, or the error stays at about 2.5, something structural is still missing.

```python
def pred_mean2(i0, dt, lv, age, code_name, typ, layout=(0,1,2,None), lstat=0, ns=32, seed=0):
    g_=torch.Generator().manual_seed(seed)
    out_=torch.tensor(S_['x'][i0,:192]); h_=torch.tensor(S_['x'][i0,192:960]).reshape(4,192)
    hf=h_*torch.exp(-dt*torch.exp(fa)); ctx=out_+LN(hf.mean(0),'core.out_norm')+MLP(torch.tensor([math.log(dt)/10],dtype=torch.float32),'gap','gelu')
    pr=MLP(ctx,'z_prior','gelu'); zs=pr[:32]+torch.exp(pr[32:])*torch.randn(ns,32,generator=g_)
    view=ctx+L_(zs,'z_head'); q=LN(view+Wo['type_emb.weight'][typ],'query_norm')
    c_=CODE[code_name]; num=torch.zeros(8); lb=0; extra=0
    if lv is not None:
        zl=zsc(code_name,'value',lv); lb=fbin(code_name,'value',lv)
        for pos,val in zip(layout,[1.0, math.log(1+age)/10, zl, 1.0]):
            if pos is not None: num[pos]=val
        extra=Wo['last_status.weight'][lstat]
    wc=q+Wo['embed.code.weight'][c_]+L_(num,'last_numeric')+Wo['embed.bins.value.weight'][lb]+extra
    lg=MLP(wc,'attr_heads.value_bin','gelu')+Wo['embed_transition.value_bin.weight'][c_*17+lb]
    p=torch.softmax(lg,-1).mean(0); cen=torch.tensor(VOCAB['value_bin_centres'][code_name],dtype=torch.float32)
    return float((p[1:]*cen).sum())
errs_t=[]
for typ in range(15):
    pm=np.array([pred_mean2(i0,dt,lv,age,'glucose_finger',typ) for (i0,dt,lv,age) in HI[:60]]); errs_t.append(round(float(np.median(np.abs(pm-obs_pm[:60]))),2))
print('glucose types', errs_t)
fm = fac[fac.event=='map_arterial'].iloc[:60]; HM=[head_inputs(r) for r in fm.itertuples()]
errs_m=[]
for typ in range(15):
    pm=np.array([pred_mean2(i0,dt,lv,age,'map_arterial',typ) for (i0,dt,lv,age) in HM]); errs_m.append(round(float(np.median(np.abs(pm-fm.predicted_mean.values))),2))
print('map types', errs_m)
```