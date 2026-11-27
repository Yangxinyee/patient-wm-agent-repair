Two fixes, age in minutes and "has" flags in slots 3 and 4, cut the error to 2.5 mg/dL overall: about 1.6 for MAP and 3.8 for finger glucose. The rest may not be a structural error. The tool averages over 4 random z draws from the prior, so its predicted means carry sampling noise that a single prior-mean z can't reproduce.

Test: compare the tool against a 64-sample average, and compare a 4-sample average against that same 64-sample reference.
- If the two errors are about equal, the replica is correct up to the tool's sampling noise.
- If the tool's error is clearly larger, something structural is still missing.

```python
def head_P(zs_list,act='relu',hasidx=(3,4),age_div=60.0,extra_logit=None):
    gpc=M2(np.log(dt)[:,None]/10.0,'gap',act); ctx=outp+hm+gpc
    zo=M2(ctx,'z_prior',act); zm=zo[:,:32]; zsd=np.exp(zo[:,32:])
    f8=np.zeros((len(fa),8)); f8[:,0]=fa.has; f8[:,1]=np.where(fa.has>0,np.log1p(fa.lastage.fillna(0)/age_div)/10,0); f8[:,2]=fa.zl
    for i in hasidx: f8[:,i]=fa.has
    base=CE[fa.code.values]+L(f8,'last_numeric')+VB[fa.lb.values]+RB_[0]
    P=0
    for eps in zs_list:
        zz=zm if eps is None else zm+zsd*eps
        q=LNf(ctx+L(zz,'z_head')+TE[1],'query_norm')
        lg=M2(q+base,'attr_heads.value_bin',act)+TT[fa.code.values*17+fa.lb.values]
        if extra_logit is not None: lg=lg+extra_logit
        lg=lg-lg.max(1,keepdims=True); p=np.exp(lg); P=P+p/p.sum(1,keepdims=True)
    return P/len(zs_list)
r_=np.random.default_rng(1)
E64=[r_.standard_normal((len(fa),32)) for _ in range(64)]
P64=head_P(E64); pm64=(P64[:,1:]*CM).sum(1)
P4a=head_P(E64[:4]); pm4a=(P4a[:,1:]*CM).sum(1)
Pm=head_P([None]); pmm=(Pm[:,1:]*CM).sum(1)
for nm,v in [('tool',obs),('4samp',pm4a),('zmean',pmm)]:
    print(nm,'MAE vs 64-avg',round(np.abs(v-pm64).mean(),3))
ob=fa.observed.values
obin=np.array([fbin(ev,v) for ev,v in zip(fa.event,ob)])
nll64=-np.log(P64[np.arange(len(fa)),obin]+1e-12)
print('NLL tool',round(fa.nll.mean(),4),'mine64',round(nll64.mean(),4),'corr',round(np.corrcoef(nll64,fa.nll)[0,1],4))
```