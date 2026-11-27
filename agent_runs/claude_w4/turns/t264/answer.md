The full layout search brings the replica down to 2.0 mg/dL RMSE. The best fit:

- query type 1;
- age in seconds, with log(1+age)/10 in column 3;
- z of the last value in column 6;
- has_last in column 0.

Part of the remaining error may come from z. The factual tool averages probabilities over 4 random prior draws, while I plug in the prior mean. Another part may come from an extra flag column I haven't placed.

Next, with the best layout I will:
1. Average the predicted distribution over 64 prior draws.
2. Separately, try adding a has flag in each of the free columns (1, 2, 4, 5, 7).

If the prior average drops the RMSE to ~1 mg/dL or less, the remainder is just the tool's own 4‑draw noise and the replica is exact enough. If an extra flag column gives a big drop instead, that column is part of the layout.

```python
b0,has,zl,lb,c=base_parts(gF,'glucose_finger',1)
A_=np.log1p(np.nan_to_num(gF.age_h.values,nan=0.)*3600.)/10*has
pre_best=b0+zl[:,None]*colp[6]+A_[:,None]*colp[3]+has[:,None]*colp[0]
for pV in [1,2,4,5,7]:
    pm=pm_from(pre_best+has[:,None]*colp[pV],lb,c,cenF); print('extra has col',pV,'rmse %.3f'%np.sqrt(np.mean((pm-yF)**2)))
# z prior averaging
def head_full(g,ev,typ,K=64,seed=0,extra=None,dout=None,dh=None):
    b=g.bi.values; out=xo[b,0:192].astype(np.float64); h=xo[b,192:960].reshape(-1,4,192).astype(np.float64)
    if dout is not None: out=out+dout
    if dh is not None: h=h+dh
    dt=g.dt_s.values; htf=h*np.exp(-dt[:,None,None]*np.exp(HW['core.flow.a']))
    gp=G(np.log(dt)[:,None]/10*HW['gap.0.weight'][:,0]+HW['gap.0.bias'])@HW['gap.2.weight'].T+HW['gap.2.bias']
    ctx=out+LNn(htf.mean(1),HW['core.out_norm.weight'],HW['core.out_norm.bias'])+gp
    zp=G(ctx@HW['z_prior.0.weight'].T+HW['z_prior.0.bias'])@HW['z_prior.2.weight'].T+HW['z_prior.2.bias']
    mu,ls=zp[:,:32],zp[:,32:]
    cc=cid[ev]; hs=np.isfinite(g.lastv.values).astype(float); lvv=np.nan_to_num(g.lastv.values,nan=0.)
    lb_=np.where(hs>0,bin_(ev,lvv),0); zl_=np.where(hs>0,zsc_(ev,lvv),0.)
    ag=np.log1p(np.nan_to_num(g.age_h.values,nan=0.)*3600.)/10*hs
    vec=np.zeros((len(g),8)); vec[:,0]=hs; vec[:,3]=ag; vec[:,6]=zl_
    rest=HW['embed.code.weight'][cc]+vec@HW['last_numeric.weight'].T+HW['last_numeric.bias']+HW['embed.bins.value.weight'][lb_]+HW['embed.bins.rate.weight'][0]
    rng=np.random.default_rng(seed); P=0
    for k in range(K):
        z=mu if K==1 else mu+np.exp(ls)*rng.standard_normal(mu.shape)
        view=ctx+z@HW['z_head.weight'].T+HW['z_head.bias']
        q=LNn(view+HW['type_emb.weight'][typ],HW['query_norm.weight'],HW['query_norm.bias'])
        lg=G((q+rest)@HW['attr_heads.value_bin.0.weight'].T+HW['attr_heads.value_bin.0.bias'])@HW['attr_heads.value_bin.2.weight'].T+HW['attr_heads.value_bin.2.bias']+HW['embed_transition.value_bin.weight'][cc*17+lb_]
        p=np.exp(lg-lg.max(1,keepdims=True)); p/=p.sum(1,keepdims=True); P=P+p/K
    return P[:,1:]@np.array(VOCAB['value_bin_centres'][ev]), P
pm1,_=head_full(gF,'glucose_finger',1,K=1); pm64,P64=head_full(gF,'glucose_finger',1,K=64)
print('K=1 rmse %.3f  K=64 rmse %.3f  sd(exp ls) mean %.3f'%(np.sqrt(np.mean((pm1-yF)**2)),np.sqrt(np.mean((pm64-yF)**2)),0))
print('resid quantiles',np.round(np.quantile(pm64-yF,[.01,.1,.5,.9,.99]),2))
```