The pipeline matched 19,451 readings across the 400 stays to their preceding state rows. That is 6,064 glucose_finger, 2,150 glucose_lab, 4,931 map_arterial and 6,306 map_cuff readings. Next I'll compute the exact head inputs for every reading (context, z prior, q, register terms, transition logits) and the reproduced predicted mean. As a sanity check against fwd_4's CSV, the selS subset should come within about 2.0 mg/dL glucose and 1.1 mmHg MAP RMS.

```python
def head_inputs(RD, X, W=Wd):
    L_ = lambda x,n: x@W[n+'.weight'].T + W[n+'.bias']
    prev = RD['prev_row'].values.astype(int); Xp_ = X[prev]
    out = Xp_[:,:192]; h = Xp_[:,192:960].reshape(-1,4,192)
    dt = (RD['minute'].values-RD['prev_min'].values)*60.0
    hf = h*np.exp(-dt[:,None]*np.exp(W['core.flow.a']))[:,None,:]
    ctx = out + ln_(hf.mean(1),W['core.out_norm.weight'],W['core.out_norm.bias']) + L_(gelu_(L_((np.log(dt)/10)[:,None],'gap.0')),'gap.2')
    zp = L_(gelu_(L_(ctx,'z_prior.0')),'z_prior.2')
    codes = RD['event'].map(cid).values; ev = RD['event'].astype(str).values
    has = RD['last_val'].notna().values.astype(float); lv=RD['last_val'].values
    age = np.where(has>0,(RD['minute'].values-RD['last_min'].values)*60.0,0.0)
    zl=np.zeros(len(RD)); lb=np.zeros(len(RD),int); ob=np.zeros(len(RD),int); cen=np.zeros((len(RD),16))
    for c in NUMV:
        s=ev==c; s2=s&(has>0); zl[s2]=zsc_(c,lv[s2]); lb[s2]=vbin(c,lv[s2]); ob[s]=vbin(c,RD['observed'].values[s]); cen[s]=CEN[c]
    f=np.zeros((len(RD),8)); f[:,0]=has; f[:,3]=np.log1p(age)/10; f[:,6]=zl
    reg = W['embed.code.weight'][codes]+f@W['last_numeric.weight'].T+W['last_numeric.bias']+W['embed.bins.value.weight'][lb]+W['embed.bins.rate.weight'][0]+W['last_status.weight'][0]
    return dict(ctx=ctx, zmu=zp[:,:32], zls=zp[:,32:], reg=reg, trans=W['embed_transition.value_bin.weight'][codes*17+lb], ob=ob, cen=cen, ev=ev)
def q_of(HI, z, W=Wd):
    return ln_(HI['ctx'] + z@W['z_head.weight'].T + W['z_head.bias'] + W['type_emb.weight'][1], W['query_norm.weight'], W['query_norm.bias'])
def head_probs(HI, W1_, b1_, W2_=None, b2_=None, nz=0, seed=0, W=Wd):
    W2_ = Wd['attr_heads.value_bin.2.weight'] if W2_ is None else W2_; b2_ = Wd['attr_heads.value_bin.2.bias'] if b2_ is None else b2_
    rng = np.random.default_rng(seed)
    zs = [HI['zmu']] if nz==0 else [HI['zmu']+np.exp(HI['zls'])*rng.standard_normal(HI['zmu'].shape) for _ in range(nz)]
    P=0
    for z in zs:
        wc = q_of(HI,z,W)+HI['reg']
        lo = gelu_(wc@W1_.T+b1_)@W2_.T+b2_+HI['trans']; lo-=lo.max(1,keepdims=True); p=np.exp(lo); p/=p.sum(1,keepdims=True); P=P+p/len(zs)
    mean=(P[:,1:]*HI['cen']).sum(1); nll=-np.log(P[np.arange(len(P)),HI['ob']])
    return mean, nll, P
HI = head_inputs(RD, XALL)
Q_all = q_of(HI, HI['zmu'])
m0, n0, P0 = head_probs(HI, W1, b1)
RD['pred0']=m0; RD['nll0']=n0
chk = RD.merge(F4[['stay_id','minute','event','predicted_mean','nll']], on=['stay_id','minute','event'])
glc = chk.event.str.startswith('glucose')
print(len(chk), 'rms g', np.sqrt(((chk.pred0-chk.predicted_mean)[glc]**2).mean()).round(2), 'rms m', np.sqrt(((chk.pred0-chk.predicted_mean)[~glc]**2).mean()).round(2))
print('mean nll (meanz) by event', RD.groupby('event')['nll0'].mean().round(3).to_dict(), 'overall', RD.nll0.mean().round(4))
```