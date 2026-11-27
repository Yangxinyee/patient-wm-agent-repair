Type 1 is confirmed, so q's type term is right and the remaining error lies in the register. One clue: removing the last-value bin embedding helped. So I'll search the layout of the 8 `last_numeric` inputs directly. The variables are the column for the last z, the column for log-age and the has-flag columns, with and without the last-bin embedding. q is precomputed once, so each layout costs only one MLP pass. A clear winner near the sampling-noise floor (about 2 mg/dL for glucose) would identify the register exactly.

```python
import itertools, time
t0=time.time()
W=Wd; prev = FR['prev_row'].values.astype(int); X = S3x[prev]
out = X[:,:192]; h = X[:,192:960].reshape(-1,4,192)
dt = (FR['minute'].values-FR['prev_min'].values)*60.0
hf = h*np.exp(-dt[:,None]*np.exp(W['core.flow.a']))[:,None,:]
L_ = lambda x,n: x@W[n+'.weight'].T + W[n+'.bias']
ctx = out + ln_(hf.mean(1),W['core.out_norm.weight'],W['core.out_norm.bias']) + L_(gelu_(L_((np.log(dt)/10)[:,None],'gap.0')),'gap.2')
zmu = L_(gelu_(L_(ctx,'z_prior.0')),'z_prior.2')[:,:32]
Q0 = ln_(ctx + zmu@W['z_head.weight'].T + W['z_head.bias'] + W['type_emb.weight'][1], W['query_norm.weight'], W['query_norm.bias'])
codes = FR['event'].map(cid).values; has = FR['last_val'].notna().values.astype(float); lv=FR['last_val'].values
age = np.where(has>0,(FR['minute'].values-FR['last_min'].values)*60.0,0.0)
zl=np.zeros(len(FR)); lb=np.zeros(len(FR),int); obb=np.zeros(len(FR),int)
for c in NUMV:
    s=(FR['event'].values==c); s2=s&(has>0); zl[s2]=zsc_(c,lv[s2]); lb[s2]=vbin(c,lv[s2]); obb[s]=vbin(c,FR['observed'].values[s])
evs=FR['event'].values; glm=np.char.startswith(evs.astype(str),'glucose'); iqr_=np.where(glm,52.0,20.0)
cen_mat = np.zeros((len(FR),16))
for c in NUMV: cen_mat[evs==c]=CEN[c]
fixed = Q0 + W['embed.code.weight'][codes] + W['last_numeric.bias'] + W['embed.bins.rate.weight'][0] + W['last_status.weight'][0]
trans = W['embed_transition.value_bin.weight'][codes*17+lb]
LW = W['last_numeric.weight']; W1=W['attr_heads.value_bin.0.weight']; b1=W['attr_heads.value_bin.0.bias']; W2=W['attr_heads.value_bin.2.weight']; b2=W['attr_heads.value_bin.2.bias']
def evalR(R):
    lo = gelu_((fixed+R)@W1.T+b1)@W2.T+b2+trans; lo-=lo.max(1,keepdims=True); p=np.exp(lo); p/=p.sum(1,keepdims=True)
    mean=(p[:,1:]*cen_mat).sum(1); e=mean-FR['predicted_mean'].values
    return np.sqrt((e[glm]**2).mean()), np.sqrt((e[~glm]**2).mean())
res=[]
la_=np.log1p(age)/10
for usebins in [0,1]:
    Bv = W['embed.bins.value.weight'][lb] if usebins else W['embed.bins.value.weight'][0][None,:]*np.ones((len(FR),1))
    for zc in range(8):
        for ac in range(8):
            if ac==zc: continue
            rest=[c for c in range(8) if c not in (zc,ac)]
            for k in [1,2]:
                for hs in itertools.combinations(rest,k):
                    R = Bv + np.outer(zl,LW[:,zc]) + np.outer(la_,LW[:,ac]) + np.outer(has,LW[:,list(hs)].sum(1))
                    rg,rm = evalR(R); res.append((rg/52+rm/20, rg, rm, usebins, zc, ac, hs))
    print('usebins',usebins,'done',round(time.time()-t0))
res.sort(key=lambda r:r[0]); print([ (round(r[1],2),round(r[2],2),r[3:]) for r in res[:8]])
```