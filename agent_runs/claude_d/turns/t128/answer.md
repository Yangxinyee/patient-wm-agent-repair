All 5,138 readings have a preceding state row, so the reproduction can be tested on the full set. Below I implement the head prediction and compare variants against the CSV. The variants are type index 1 vs 3 and gap input log(dt)/10 vs log(dt), with the z prior mean and the default register layout `[has_last, age, z_last, has_value, …]`. I'll print RMS error in the predicted mean and mean absolute NLL difference, so a correct variant should stand out clearly.

```python
def predict(FR, typ=3, gap_scale=10.0, age_div=1.0, side='left', nz=0, lc=(0,1,2,3), status_idx=0, seed=0, W=None, ret_hidden=False):
    W = Wd if W is None else W
    L_ = lambda x,n: x@W[n+'.weight'].T + W[n+'.bias']
    prev = FR['prev_row'].values.astype(int); X = S3x[prev]
    out = X[:,:192]; h = X[:,192:960].reshape(-1,4,192)
    dt = (FR['minute'].values-FR['prev_min'].values)*60.0
    hf = h*np.exp(-dt[:,None]*np.exp(W['core.flow.a']))[:,None,:]
    m = ln_(hf.mean(1), W['core.out_norm.weight'], W['core.out_norm.bias'])
    g = L_(gelu_(L_((np.log(dt)/gap_scale)[:,None],'gap.0')),'gap.2')
    ctx = out+m+g
    zp = L_(gelu_(L_(ctx,'z_prior.0')),'z_prior.2'); zmu, zls = zp[:,:32], zp[:,32:]
    codes = FR['event'].map(cid).values
    has = FR['last_val'].notna().values.astype(float)
    lv = FR['last_val'].values; age = np.where(has>0,(FR['minute'].values-FR['last_min'].values)*60.0/age_div,0.0)
    zl = np.zeros(len(FR)); lb = np.zeros(len(FR),int)
    for c in NUMV:
        s = (FR['event'].values==c)&(has>0)
        zl[s]=zsc_(c,lv[s]); lb[s]=vbin(c,lv[s],side)
    f = np.zeros((len(FR),8)); f[:,lc[0]]=has; f[:,lc[1]]=np.log1p(age)/10; f[:,lc[2]]=zl; f[:,lc[3]]=has
    reg = W['embed.code.weight'][codes] + f@W['last_numeric.weight'].T + W['last_numeric.bias'] + W['embed.bins.value.weight'][lb] + W['embed.bins.rate.weight'][0] + W['last_status.weight'][status_idx]
    trans = W['embed_transition.value_bin.weight'][codes*17+lb]
    rng = np.random.default_rng(seed); zs = [zmu] if nz==0 else [zmu+np.exp(zls)*rng.standard_normal(zmu.shape) for _ in range(nz)]
    P = 0
    for z in zs:
        view = ctx + z@W['z_head.weight'].T + W['z_head.bias']
        q = ln_(view+W['type_emb.weight'][typ], W['query_norm.weight'], W['query_norm.bias'])
        hid = gelu_(L_(q+reg,'attr_heads.value_bin.0'))
        lo = hid@W['attr_heads.value_bin.2.weight'].T + W['attr_heads.value_bin.2.bias'] + trans
        lo = lo-lo.max(1,keepdims=True); p = np.exp(lo); p/=p.sum(1,keepdims=True); P = P+p/len(zs)
    mean = np.zeros(len(FR)); nll=np.zeros(len(FR))
    for c in NUMV:
        s = FR['event'].values==c
        mean[s] = P[s,1:]@CEN[c]
        ob = vbin(c,FR['observed'].values[s],side); nll[s] = -np.log(P[s,ob])
    return (mean,nll,P) if not ret_hidden else (mean,nll,P,hid)
def score(res,tag):
    mean,nll=res[0],res[1]; e = mean-FR['predicted_mean'].values; dn = nll-FR['nll'].values
    gl_ = FR['event'].str.startswith('glucose').values
    print(f"{tag}: rms_g {np.sqrt((e[gl_]**2).mean()):.2f} rms_m {np.sqrt((e[~gl_]**2).mean()):.2f} | dnll {np.abs(dn).mean():.3f} bias {dn.mean():+.3f}")
for typ in [1,3]:
    for gs in [10.0,1.0]:
        score(predict(FR,typ=typ,gap_scale=gs), f"typ{typ} gs{gs}")
```