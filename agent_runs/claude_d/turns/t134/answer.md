Dropping the last-value bin embedding helps (glucose RMS 5.7, MAP 1.9), but the reproduction is still off by about 0.11 IQR for both glucose and MAP. A relative error that similar suggests something global in the view or q, rather than the register. Below I test view variants: no time flow, without `out`, without the h term, and time units for the gap and the flow.

```python
def predict_view(FR, mode, W=None):
    W = Wd if W is None else W
    L_ = lambda x,n: x@W[n+'.weight'].T + W[n+'.bias']
    prev = FR['prev_row'].values.astype(int); X = S3x[prev]
    out = X[:,:192]; h = X[:,192:960].reshape(-1,4,192)
    dt = (FR['minute'].values-FR['prev_min'].values)*60.0
    dtf = dt/60 if mode=='flow_min' else dt
    hf = h if mode=='noflow' else h*np.exp(-dtf[:,None]*np.exp(W['core.flow.a']))[:,None,:]
    m = ln_(hf.mean(1), W['core.out_norm.weight'], W['core.out_norm.bias'])
    dtg = dt/60 if mode=='gap_min' else (dt/3600 if mode=='gap_h' else dt)
    g = L_(gelu_(L_((np.log(dtg)/10)[:,None],'gap.0')),'gap.2')
    ctx = (0 if mode=='noout' else out) + (0 if mode=='nom' else m) + g
    zp = L_(gelu_(L_(ctx,'z_prior.0')),'z_prior.2'); zmu = zp[:,:32]
    view = ctx + zmu@W['z_head.weight'].T + W['z_head.bias']
    return view
_orig_predict = predict
def score_view(mode):
    v = predict_view(FR, mode)
    # reuse predict by temporarily injecting view: recompute downstream
    W=Wd; codes = FR['event'].map(cid).values
    has = FR['last_val'].notna().values.astype(float); lv=FR['last_val'].values
    age = np.where(has>0,(FR['minute'].values-FR['last_min'].values)*60.0,0.0)
    zl=np.zeros(len(FR)); lb=np.zeros(len(FR),int)
    for c in NUMV:
        s=(FR['event'].values==c)&(has>0); zl[s]=zsc_(c,lv[s]); lb[s]=vbin(c,lv[s])
    f=np.zeros((len(FR),8)); f[:,0]=has; f[:,1]=np.log1p(age)/10; f[:,2]=zl; f[:,3]=has
    reg = W['embed.code.weight'][codes]+f@W['last_numeric.weight'].T+W['last_numeric.bias']+W['embed.bins.value.weight'][lb]+W['embed.bins.rate.weight'][0]+W['last_status.weight'][0]
    q = ln_(v+W['type_emb.weight'][1], W['query_norm.weight'], W['query_norm.bias'])
    lo = gelu_((q+reg)@W['attr_heads.value_bin.0.weight'].T+W['attr_heads.value_bin.0.bias'])@W['attr_heads.value_bin.2.weight'].T+W['attr_heads.value_bin.2.bias']+W['embed_transition.value_bin.weight'][codes*17+lb]
    lo-=lo.max(1,keepdims=True); p=np.exp(lo); p/=p.sum(1,keepdims=True)
    mean=np.zeros(len(FR)); nll=np.zeros(len(FR))
    for c in NUMV:
        s=FR['event'].values==c; mean[s]=p[s,1:]@CEN[c]; nll[s]=-np.log(p[s,vbin(c,FR['observed'].values[s])])
    score((mean,nll),mode)
for mode in ['base','noflow','flow_min','gap_min','gap_h','noout','nom']: score_view(mode)
```