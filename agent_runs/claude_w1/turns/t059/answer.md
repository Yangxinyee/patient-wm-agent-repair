The error is spread fairly evenly across codes, hours and gap lengths, so no single subgroup explains it. One concrete lead: the "last status" table has an all-zero row 0 and a large row 1 (norm 13.2), but my replica never adds row 1. If the register adds status row 1 whenever a previous value exists, that alone could account for much of the mismatch. I'll test that, with "always" as a fallback and both activations. A sharp drop in MAE towards the ~1.2 sampling-noise floor would confirm it.

```python
def head_P2(zs_list,act='relu',hasidx=(3,4),age_div=60.0,ls_mode='has',extra_logit=None,ret_hidden=False):
    gpc=M2(np.log(dt)[:,None]/10.0,'gap',act); ctx=outp+hm+gpc
    zo=M2(ctx,'z_prior',act); zm=zo[:,:32]; zsd=np.exp(zo[:,32:])
    f8=np.zeros((len(fa),8)); f8[:,0]=fa.has; f8[:,1]=np.where(fa.has>0,np.log1p(fa.lastage.fillna(0)/age_div)/10,0); f8[:,2]=fa.zl
    for i in hasidx: f8[:,i]=fa.has
    base=CE[fa.code.values]+L(f8,'last_numeric')+VB[fa.lb.values]
    if ls_mode=='has': base=base+fa.has.values[:,None]*LS[1]
    elif ls_mode=='always': base=base+LS[1]
    P=0
    for eps in zs_list:
        zz=zm if eps is None else zm+zsd*eps
        q=LNf(ctx+L(zz,'z_head')+TE[1],'query_norm')
        lg=M2(q+base,'attr_heads.value_bin',act)+TT[fa.code.values*17+fa.lb.values]
        if extra_logit is not None: lg=lg+extra_logit
        lg=lg-lg.max(1,keepdims=True); p=np.exp(lg); P=P+p/p.sum(1,keepdims=True)
    return P/len(zs_list)
for act in ['relu','gelu']:
  for lsm in ['none','has','always']:
    for hasidx in [(3,4),(3,)]:
        P16=head_P2(E64[:16],act=act,ls_mode=lsm,hasidx=hasidx); pm16=(P16[:,1:]*CM).sum(1)
        nll16=-np.log(P16[np.arange(len(fa)),obin]+1e-12)
        print(act,lsm,hasidx,'MAE',round(np.abs(pm16-obs).mean(),3),'NLL',round(nll16.mean(),4),'corr',round(np.corrcoef(nll16,fa.nll)[0,1],3))
```