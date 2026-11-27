The vocabulary pins down the remaining head details:

- the predicted mean is Σ_{b=1..16} p(b)·centre[b−1], with no renormalisation;
- z‑scores are (x − median)/IQR, clipped to ±10;
- a fine bin is 1 + the number of edges below the value;
- the flow decay is one 192‑vector shared by all slots.

Two details remain unknown: the 8‑input layout of `last_numeric` and the type index of a value query. I'll find both empirically.

Next I'll build the value‑head replica:
- `view` = out + out_norm(mean decayed slots) + gap(log dt/10) + z_head(z), with z at the prior mean;
- query_norm(view + type_emb[t]) plus the code embedding and register terms;
- the 192→192→17 MLP plus the transition table, then the bin‑centre mean.

I'll grid‑search the unknown choices against fwd_2's predicted finger means, using the exact fwd_7‑original states.
- If the best combination reaches RMSE well under 1 mg/dL, the replica is trustworthy and I can take gradients from it.
- If nothing gets below a few mg/dL, a structural piece is wrong.

```python
from scipy.special import erf
def G(x): return 0.5*x*(1+erf(x/np.sqrt(2)))
def LNn(x,w,b,eps=1e-5):
    m=x.mean(-1,keepdims=True); v=((x-m)**2).mean(-1,keepdims=True); return (x-m)/np.sqrt(v+eps)*w+b
Wt=lambda n: np.asarray(load_original(n),dtype=np.float64)
HW={n:Wt(n) for n in ['core.flow.a','core.out_norm.weight','core.out_norm.bias','gap.0.weight','gap.0.bias','gap.2.weight','gap.2.bias','z_prior.0.weight','z_prior.0.bias','z_prior.2.weight','z_prior.2.bias','z_head.weight','z_head.bias','query_norm.weight','query_norm.bias','type_emb.weight','embed.code.weight','last_numeric.weight','last_numeric.bias','last_status.weight','embed.bins.value.weight','embed.bins.rate.weight','attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias','embed_transition.value_bin.weight']}
print('last_numeric col norms',np.round(np.linalg.norm(HW['last_numeric.weight'],axis=0),3))
num={(e['code'],e['attribute']):e for e in VOCAB['numeric']}; cid={c['name']:c['id'] for c in VOCAB['codes']}
def zsc_(code,x): e=num[(code,'value')]; return np.clip((x-e['median'])/e['iqr'],-10,10)
def bin_(code,x): e=np.array(num[(code,'value')]['fine_bin_edges']); return 1+(e[None,:]<np.asarray(x)[:,None]).sum(1)
def prep(ev):
    g=fc[(fc.event==ev)&(fc.bi>=0)].copy(); lv=[];la=[]
    for s_,gg in fc[fc.event==ev].groupby('stay_id'):
        hh=gg.hour.values; vv=gg.observed.values
        for j,ix in enumerate(gg.index):
            p_=np.where(hh<hh[j])[0]
            lv.append((ix,vv[p_[-1]] if len(p_) else np.nan,(hh[j]-hh[p_[-1]]) if len(p_) else np.nan))
    L=pd.DataFrame(lv,columns=['ix','lastv','age_h']).set_index('ix'); g=g.join(L)
    g['dt_s']=(g.hour.values-hr[g.bi.values])*3600.; return g
def head_pred(g,ev,typ,lay,stat,ageu,z=None):
    b=g.bi.values; out=xo[b,0:192].astype(np.float64); h=xo[b,192:960].reshape(-1,4,192).astype(np.float64)
    dt=g.dt_s.values; htf=h*np.exp(-dt[:,None,None]*np.exp(HW['core.flow.a']))
    gp=G(np.log(dt)[:,None]/10*HW['gap.0.weight'][:,0]+HW['gap.0.bias'])@HW['gap.2.weight'].T+HW['gap.2.bias']
    ctx=out+LNn(htf.mean(1),HW['core.out_norm.weight'],HW['core.out_norm.bias'])+gp
    zp=G(ctx@HW['z_prior.0.weight'].T+HW['z_prior.0.bias'])@HW['z_prior.2.weight'].T+HW['z_prior.2.bias']
    zz=zp[:,:32] if z is None else z
    view=ctx+zz@HW['z_head.weight'].T+HW['z_head.bias']
    q=LNn(view+HW['type_emb.weight'][typ],HW['query_norm.weight'],HW['query_norm.bias'])
    c=cid[ev]; has=np.isfinite(g.lastv.values).astype(float); lvv=np.nan_to_num(g.lastv.values,nan=0.)
    lb=np.where(has>0,bin_(ev,lvv),0); zl=np.where(has>0,zsc_(ev,lvv),0.)
    age=np.nan_to_num(g.age_h.values,nan=0.)*(3600. if ageu=='s' else 1.)
    vec=np.zeros((len(g),8)); vec[:,0]=has; vec[:,1]=np.log1p(age)/10*has; vec[:,2]=zl
    if lay=='A': vec[:,3]=has
    wc=q+HW['embed.code.weight'][c]+vec@HW['last_numeric.weight'].T+HW['last_numeric.bias']+HW['embed.bins.value.weight'][lb]+HW['embed.bins.rate.weight'][0]
    if stat is not None: wc=wc+HW['last_status.weight'][stat]
    lg=G(wc@HW['attr_heads.value_bin.0.weight'].T+HW['attr_heads.value_bin.0.bias'])@HW['attr_heads.value_bin.2.weight'].T+HW['attr_heads.value_bin.2.bias']+HW['embed_transition.value_bin.weight'][c*17+lb]
    p=np.exp(lg-lg.max(1,keepdims=True)); p/=p.sum(1,keepdims=True)
    cen=np.array(VOCAB['value_bin_centres'][ev]); return p[:,1:]@cen, p
gF=prep('glucose_finger'); res=[]
for typ in range(15):
    for lay in 'AB':
        for stat in [None,0,1]:
            for ageu in 'sh':
                pm,_=head_pred(gF,'glucose_finger',typ,lay,stat,ageu); res.append((np.sqrt(np.mean((pm-gF.predicted_mean.values)**2)),typ,lay,stat,ageu))
res.sort(key=lambda r:r[0]); print(res[:8])
```