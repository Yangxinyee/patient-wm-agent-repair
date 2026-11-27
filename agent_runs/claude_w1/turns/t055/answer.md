The previous run died before `zsc` and `fbin` were defined, so this run failed on them. Within-stay order is confirmed monotone. I'll define the two helpers and continue from where the register features are built.

```python
def zsc(ev,x): e=num[(ev,'value')]; return np.clip((x-e['median'])/e['iqr'],-10,10)
def fbin(ev,x): e=num[(ev,'value')]; return 1+np.searchsorted(np.array(e['fine_bin_edges']),x,side='left')
fa['has']=fa.lastv.notna().astype(float)
fa['zl']=[zsc(ev,v) if v==v else 0.0 for ev,v in zip(fa.event,fa.lastv)]
fa['lb']=[int(fbin(ev,v)) if v==v else 0 for ev,v in zip(fa.event,fa.lastv)]
fa['la']=np.where(fa.has>0,np.log1p(fa.lastage.fillna(0))/10,0.0)
Xp=X_[fa.prow.values]; outp=Xp[:,:192]; hp=Xp[:,192:960].reshape(-1,4,192)
dt=(fa.hour.values-hr_[fa.prow.values])*3600
dec=np.exp(-dt[:,None]*np.exp(Wt('core.flow.a'))[None,:])
hm=LNf((hp*dec[:,None,:]).mean(1),'core.out_norm')
cent={k:np.array(v) for k,v in vc.items()}
CM=np.stack([cent[ev] for ev in fa.event]); obs=fa.predicted_mean.values
TT=Wt('embed_transition.value_bin.weight'); TE=Wt('type_emb.weight'); CE=Wt('embed.code.weight')
VB=Wt('embed.bins.value.weight'); RB_=Wt('embed.bins.rate.weight'); LS=Wt('last_status.weight')
res=[]
for act in ['relu','gelu']:
  for gs in [10.0,1.0]:
    gp=M2(np.log(dt)[:,None]/gs,'gap',act); ctx=outp+hm+gp
    zpm=M2(ctx,'z_prior',act)[:,:32]; view=ctx+L(zpm,'z_head')
    for lnv in ['A','B']:
        f8=np.zeros((len(fa),8)); f8[:,0]=fa.has; f8[:,1]=fa.la; f8[:,2]=fa.zl
        if lnv=='B': f8[:,3]=fa.has
        lnum=L(f8,'last_numeric')
        for st in [None,0,1]:
          base=CE[fa.code.values]+lnum+VB[fa.lb.values]+RB_[0]+(0 if st is None else LS[st])
          for ty in range(15):
            q=LNf(view+TE[ty],'query_norm')
            lg=M2(q+base,'attr_heads.value_bin',act)+TT[fa.code.values*17+fa.lb.values]
            lg=lg-lg.max(1,keepdims=True); p=np.exp(lg); p/=p.sum(1,keepdims=True)
            pm=(p[:,1:]*CM).sum(1)
            res.append((round(np.abs(pm-obs).mean(),3),act,gs,lnv,st,ty))
res.sort(key=lambda r:r[0]); print(len(fa)); print(res[:6])
```