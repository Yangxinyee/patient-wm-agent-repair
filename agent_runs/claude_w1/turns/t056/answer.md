The replica is close but not exact: mean absolute error 3.4 against the tool's predicted means. The best variant uses:
- type row 1, which fits type = atom_kind × 5 + event_kind;
- gap input log(dt)/10;
- a "has value" flag in register slot 3.

ReLU and GELU fit about equally well. Getting the register layout right cut the error from 5.7 to 3.4, so the remaining gap probably comes from three things:
- the rest of the 8-feature register layout or the age units;
- using the prior mean of z instead of the tool's 4 sampled z;
- the gap scaling in the view.

I'll search those jointly. I'll score per-code MAE and also the NLL of the observed bin, which checks the whole predicted distribution and not just its mean. If one variant drops below about 0.5 mg/dL MAE, the replica is good enough to design an edit on.

```python
def run_head(act='relu',gs_ctx=10.0,gs_view=10.0,ty=1,hasidx=(3,),age_div=1.0,zmode='mean',ret=False,extra=None):
    gpc=M2(np.log(dt)[:,None]/gs_ctx,'gap',act); ctx=outp+hm+gpc
    zo=M2(ctx,'z_prior',act); zm=zo[:,:32]; zs=np.exp(zo[:,32:])
    gpv=M2(np.log(dt)[:,None]/gs_view,'gap',act); vbase=outp+hm+gpv
    f8=np.zeros((len(fa),8)); f8[:,0]=fa.has; f8[:,1]=np.where(fa.has>0,np.log1p(fa.lastage.fillna(0)/age_div)/10,0); f8[:,2]=fa.zl
    for i in hasidx: f8[:,i]=fa.has
    base=CE[fa.code.values]+L(f8,'last_numeric')+VB[fa.lb.values]+RB_[0]
    if extra is not None: base=base+extra
    rng_=np.random.default_rng(0)
    zlist=[zm] if zmode=='mean' else [zm+zs*rng_.standard_normal(zm.shape) for _ in range(4)]
    P=0
    for zz in zlist:
        q=LNf(vbase+L(zz,'z_head')+TE[ty],'query_norm')
        lg=M2(q+base,'attr_heads.value_bin',act)+TT[fa.code.values*17+fa.lb.values]
        lg=lg-lg.max(1,keepdims=True); p=np.exp(lg); P=P+p/p.sum(1,keepdims=True)
    P=P/len(zlist)
    pm=(P[:,1:]*CM).sum(1)
    if ret: return P,pm
    return np.abs(pm-obs).mean()
res=[]
for hasidx in [(3,),(3,5),(3,5,7),(4,),(3,4),(5,),(6,),(7,)]:
  for age_div in [1.0,60.0,3600.0]:
    for gsv in [10.0,1.0]:
        res.append((round(run_head(hasidx=hasidx,age_div=age_div,gs_view=gsv),3),hasidx,age_div,gsv))
res.sort(key=lambda r:r[0]); print(res[:5])
b=res[0]
print('zmode samples:',round(run_head(hasidx=b[1],age_div=b[2],gs_view=b[3],zmode='samp'),3))
P_,pm_=run_head(hasidx=b[1],age_div=b[2],gs_view=b[3],ret=True)
fa['pm_rep']=pm_
print(fa.assign(e=np.abs(pm_-obs)).groupby('event').e.agg(['mean','median']).round(3))
```