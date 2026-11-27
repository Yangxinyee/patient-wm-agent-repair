The table has 1,887 readings from the 38 shared stays. The gap to the previous bundle is usually 55 minutes, consistent with the drug or END bundle at h−1 plus 5 minutes.

Next I will build the head and grid-search its unknowns:
- the gap input scaling (3 options);
- the `last_numeric` input layout (2 options);
- whether `last_status[0]` is added;
- the type index (15 options).

For each combination I will compare predicted means with the tool's, using the prior mean for z for now. I will also print a few row and column norms of tensors I am unsure are used. Success means a configuration with MAE well under 1 unit. A best MAE of several units would point to a structural error, for example in how the state or view is assembled.

```python
CID={'glucose_finger':6,'map_arterial':7,'map_cuff':5,'glucose_lab':10}
print("norms: bins.rate[0]",np.linalg.norm(Wt['embed.bins.rate.weight'][0]).round(3),"bins.value[0]",np.linalg.norm(Wt['embed.bins.value.weight'][0]).round(3),
      "last_status rows",np.linalg.norm(Wt['last_status.weight'],axis=1).round(3),"last_numeric cols",np.linalg.norm(Wt['last_numeric.weight'],axis=0).round(2))
Xs=S2['x'].astype(np.float64)
def view_of(k,dt,gapmode,zmode='mean',rng=None):
    st=Xs[k]; out=st[:,:192]; hs=st[:,192:960].reshape(-1,4,192)
    hf=hs*np.exp(-dt[:,None,None]*np.exp(Wt['core.flow.a'])[None,None,:])
    m=lnorm(hf.mean(1),Wt['core.out_norm.weight'],Wt['core.out_norm.bias'])
    gx={'log':np.log(dt),'log10':np.log(dt)/10,'log1p10':np.log1p(dt)/10}[gapmode][:,None]
    g=L(gelu(L(gx,'gap.0')),'gap.2'); ctx=out+m+g
    pr=L(gelu(L(ctx,'z_prior.0')),'z_prior.2'); z=pr[:,:32]
    return ctx+L(z,'z_head')
def regs(F,layout):
    has=F.prev_val.notna().values.astype(float); age=np.where(has>0,(F.hour-F.prev_hr).values*3600,0)
    zl=np.zeros(len(F)); lb=np.zeros(len(F),int)
    for ev in CID:
        mk=(F.event==ev).values&(has>0); zl[mk]=zval(ev,F.prev_val.values[mk]); lb[mk]=fbin(ev,F.prev_val.values[mk])
    la=np.log1p(age)/10
    if layout=='A': R=np.column_stack([has,la,zl,np.zeros((len(F),5))])
    else: R=np.column_stack([has,la,zl,has,np.zeros((len(F),4))])
    return R,lb
def head_mean(F,V,T,layout,status,code_ids):
    R,lb=regs(F,layout)
    q=lnorm(V+Wt['type_emb.weight'][T],Wt['query_norm.weight'],Wt['query_norm.bias'])
    wc=q+Wt['embed.code.weight'][code_ids]+L(R,'last_numeric')+Wt['embed.bins.value.weight'][lb]+Wt['embed.bins.rate.weight'][0]
    if status: wc=wc+Wt['last_status.weight'][0]
    lg=L(gelu(L(wc,'attr_heads.value_bin.0')),'attr_heads.value_bin.2')+Wt['embed_transition.value_bin.weight'][code_ids*17+lb]
    p=np.exp(lg-lg.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
    cen=np.stack([CEN[e] for e in F.event]); return (p[:,1:]*cen).sum(1),p
code_ids=F.event.map(CID).values; kv=F.k.values; dtv=F.dt.values.astype(float)
best=[]
for gm in ['log','log10','log1p10']:
    V=view_of(kv,dtv,gm)
    for lay in ['A','B']:
        for stt in [False,True]:
            for T in range(15):
                pm,_=head_mean(F,V,T,lay,stt,code_ids); best.append((np.abs(pm-F.predicted_mean.values).mean(),gm,lay,stt,T))
best.sort(); print(best[:6])
```