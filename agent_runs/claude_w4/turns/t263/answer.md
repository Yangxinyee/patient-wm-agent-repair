The replica gets the overall structure right but not the details. The best combination has an RMSE of 7.0 mg/dL against the model's own predicted means: type index 1, age in hours, and `has` flags at columns 0 and 3. All 8 columns of `last_numeric` carry similar weight, so all 8 inputs are in use. `last_status[0]` is a zero row, so measurements effectively ignore it.

The most likely culprit is my guess at the 8‑input layout: which position holds the z‑scored last value, the log‑age and the "has" flags, and in what time unit.

Next I'll search all placements of the four register features (has_last, log(1+age)/10, z_last, has_value) over the 8 columns, with age in seconds, minutes or hours, for type indices 1, 7 and 9. The search precomputes the layout‑independent part of the MLP input, so it is cheap.
- If some layout brings the RMSE below ~1 mg/dL, the replica is usable for gradients.
- If the floor stays around 5–7, the error comes from elsewhere: the z draws, the gap scaling, or within‑bundle conditioning.

```python
import itertools
def base_parts(g,ev,typ):
    b=g.bi.values; out=xo[b,0:192].astype(np.float64); h=xo[b,192:960].reshape(-1,4,192).astype(np.float64)
    dt=g.dt_s.values; htf=h*np.exp(-dt[:,None,None]*np.exp(HW['core.flow.a']))
    gp=G(np.log(dt)[:,None]/10*HW['gap.0.weight'][:,0]+HW['gap.0.bias'])@HW['gap.2.weight'].T+HW['gap.2.bias']
    ctx=out+LNn(htf.mean(1),HW['core.out_norm.weight'],HW['core.out_norm.bias'])+gp
    zp=G(ctx@HW['z_prior.0.weight'].T+HW['z_prior.0.bias'])@HW['z_prior.2.weight'].T+HW['z_prior.2.bias']
    view=ctx+zp[:,:32]@HW['z_head.weight'].T+HW['z_head.bias']
    q=LNn(view+HW['type_emb.weight'][typ],HW['query_norm.weight'],HW['query_norm.bias'])
    c=cid[ev]; has=np.isfinite(g.lastv.values).astype(float); lvv=np.nan_to_num(g.lastv.values,nan=0.)
    lb=np.where(has>0,bin_(ev,lvv),0); zl=np.where(has>0,zsc_(ev,lvv),0.)
    base=q+HW['embed.code.weight'][c]+HW['last_numeric.bias']+HW['embed.bins.value.weight'][lb]+HW['embed.bins.rate.weight'][0]
    return base@HW['attr_heads.value_bin.0.weight'].T+HW['attr_heads.value_bin.0.bias'], has, zl, lb, c
colp=HW['last_numeric.weight'].T@HW['attr_heads.value_bin.0.weight'].T   # (8,192)
cenF=np.array(VOCAB['value_bin_centres']['glucose_finger'])
def pm_from(pre,lb,c,cen):
    lg=G(pre)@HW['attr_heads.value_bin.2.weight'].T+HW['attr_heads.value_bin.2.bias']+HW['embed_transition.value_bin.weight'][c*17+lb]
    p=np.exp(lg-lg.max(1,keepdims=True)); p/=p.sum(1,keepdims=True); return p[:,1:]@cen
yF=gF.predicted_mean.values; res=[]
for typ in [1,7,9]:
    b0,has,zl,lb,c=base_parts(gF,'glucose_finger',typ)
    for ageu,sc in [('s',3600.),('m',60.),('h',1.)]:
        A_=np.log1p(np.nan_to_num(gF.age_h.values,nan=0.)*sc)/10*has
        for pZ,pA,pH in itertools.permutations(range(8),3):
            pre0=b0+zl[:,None]*colp[pZ]+A_[:,None]*colp[pA]+has[:,None]*colp[pH]
            for pV in [None]+[j for j in range(8) if j not in (pZ,pA,pH)]:
                pre=pre0 if pV is None else pre0+has[:,None]*colp[pV]
                pm=pm_from(pre,lb,c,cenF); res.append((np.sqrt(np.mean((pm-yF)**2)),typ,ageu,pZ,pA,pH,pV))
res.sort(key=lambda r:r[0]); print(len(res)); print(res[:10])
```