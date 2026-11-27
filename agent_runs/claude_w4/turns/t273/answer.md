`step_cv` doesn't return the same shape as `step_full`: its first output has only 768 numbers. It probably packs its outputs differently, maybe new slots for a single row or summary statistics.

Rather than guess again, I'll print the type and shape of every element it returns. I'll then automatically pick the (2319, 4, 192) new‑slot and (2319, 192) `out` arrays if they are there, and score them against the exact states. The head type search for the other three readings is independent of the step replica, so it runs in the same go.

If `'ageKV'` gives R² ≥ 0.99, the Jacobian comes next. If its outputs aren't per‑row states, I'll reconstruct the variant from `step_full`'s pieces or switch to empirical probing.

```python
rb=step_cv(np.arange(2319),torch.tensor(Ep,dtype=torch.float32),Ht,'ageKV')
els=list(rb) if isinstance(rb,(tuple,list)) else [rb]
print(type(rb),[(type(e).__name__,tuple(getattr(e,'shape',())) if hasattr(e,'shape') else e) for e in els][:8])
Y_=[A(e) for e in els if hasattr(e,'shape') and A(e).size==2319*768]; O_=[A(e) for e in els if hasattr(e,'shape') and A(e).size==2319*192]
if Y_ and O_:
    y0=Y_[0].reshape(2319,4,192); o0=O_[0].reshape(2319,192); print('ageKV R2 y %.5f out %.5f relerr y %.4f'%(r2f(y0,yt_),r2f(o0,ot_),np.linalg.norm(y0-yt_)/np.linalg.norm(yt_)))
fa_=HW['core.flow.a']
def reg_rest(g,ev):
    cc=cid[ev]; hs=np.isfinite(g.lastv.values).astype(float); lvv=np.nan_to_num(g.lastv.values,nan=0.)
    lb_=np.where(hs>0,bin_(ev,lvv),0); zl_=np.where(hs>0,zsc_(ev,lvv),0.)
    ag=np.log1p(np.nan_to_num(g.age_h.values,nan=0.)*3600.)/10*hs
    vec=np.zeros((len(g),8)); vec[:,0]=hs; vec[:,3]=ag; vec[:,6]=zl_
    return HW['embed.code.weight'][cc]+vec@HW['last_numeric.weight'].T+HW['last_numeric.bias']+HW['embed.bins.value.weight'][lb_]+HW['embed.bins.rate.weight'][0], HW['embed_transition.value_bin.weight'][cc*17+lb_]
def head_at(rest,trans,cen,typ,OUT,HM,dt):
    htfm=HM*np.exp(-dt[:,None]*np.exp(fa_))
    gp=G(np.log(dt)[:,None]/10*HW['gap.0.weight'][:,0]+HW['gap.0.bias'])@HW['gap.2.weight'].T+HW['gap.2.bias']
    ctx=OUT+LNn(htfm,HW['core.out_norm.weight'],HW['core.out_norm.bias'])+gp
    zp=G(ctx@HW['z_prior.0.weight'].T+HW['z_prior.0.bias'])@HW['z_prior.2.weight'].T+HW['z_prior.2.bias']
    view=ctx+zp[:,:32]@HW['z_head.weight'].T+HW['z_head.bias']
    q=LNn(view+HW['type_emb.weight'][typ],HW['query_norm.weight'],HW['query_norm.bias'])
    lg=G((q+rest)@HW['attr_heads.value_bin.0.weight'].T+HW['attr_heads.value_bin.0.bias'])@HW['attr_heads.value_bin.2.weight'].T+HW['attr_heads.value_bin.2.bias']+trans
    p=np.exp(lg-lg.max(1,keepdims=True)); p/=p.sum(1,keepdims=True); return p[:,1:]@cen
EVS=['glucose_finger','glucose_lab','map_arterial','map_cuff']; HD={}
for ev in EVS:
    g=prep(ev); g=g[g.bi.isin(list(pos.keys()))].copy(); rest,trans=reg_rest(g,ev); cen=np.array(VOCAB['value_bin_centres'][ev])
    OUTb=xo[g.bi.values,0:192].astype(np.float64); HMb=xo[g.bi.values,192:960].reshape(-1,4,192).mean(1).astype(np.float64); dt=g.dt_s.values
    best=min(((np.sqrt(np.mean((head_at(rest,trans,cen,t_,OUTb,HMb,dt)-g.predicted_mean.values)**2)),t_) for t_ in range(15)))
    HD[ev]=dict(g=g,rest=rest,trans=trans,cen=cen,typ=best[1],OUT=OUTb,HM=HMb,dt=dt,kk=np.array([pos[b] for b in g.bi.values]))
    print(ev,'best type',best[1],'rmse %.3f'%best[0],'n',len(g),'pred sd %.1f'%g.predicted_mean.std())
```