The diagnostics separate the two possibilities. My inputs carry the right information: a linear map from [ctx, e, u] explains 84% of the stored z variance. The per-dimension correlation of my MLP output is only about 0.56, though. So the encoders and context are roughly right, but some detail is wrong.

Next I'll split the error by bundle type: reading-only, drug-only, mixed, and END/visit-only bundles. I'll also try two context variants: h without the time decay, and context without `out`.
- Small error on reading-only bundles would clear the measurement encoder and context, putting the fault on the action side, such as END atoms or rate and dose handling.
- Uniformly large errors point to the context.

```python
btype=[];errA=[];errB=[];errC=[]
idx=0
for s_ in list(np.unique(sid_))[:12]:
    ii=starts[s_]; B=bund[s_]
    for k in range(1,len(ii)):
        j0,j1=ii[k-1],ii[k]
        hasm=any(a['ev'] not in ACTION and a['kind']==0 for a in B[k]); hasa=any(a['ev'] in ACTION for a in B[k])
        btype.append(('M' if hasm else '')+('A' if hasa else '')+('E' if any(a['kind']==2 for a in B[k]) else ''))
        idx+=1
btype=np.array(btype)
def relerr(zp): return np.abs(zp-ZT).mean(1)/np.abs(ZT).mean()
gp=M2(GI[:,None],'gap','gelu')
variants={}
variants['base']=M2(np.concatenate([CI+gp,EI,UI],1),'z_post','gelu')[:,:32]
variants['nogap']=M2(np.concatenate([CI,EI,UI],1),'z_post','gelu')[:,:32]
# ctx without decay
CI_nd=[];
for s_ in list(np.unique(sid_))[:12]:
    ii=starts[s_]
    for k in range(1,len(ii)):
        j0=ii[k-1]; CI_nd.append(X_[j0,:192]+LNf(X_[j0,192:960].reshape(4,192).mean(0),'core.out_norm'))
CI_nd=np.array(CI_nd)
variants['nodecay']=M2(np.concatenate([CI_nd+gp,EI,UI],1),'z_post','gelu')[:,:32]
variants['noout']=M2(np.concatenate([CI-X_[[0]*0+[ii for ii in []] ] if False else CI- np.array([X_[starts[s][k-1],:192] for s in list(np.unique(sid_))[:12] for k in range(1,len(starts[s]))])+gp,EI,UI],1),'z_post','gelu')[:,:32]
for nm,zp in variants.items():
    re=relerr(zp)
    print(nm, {t:round(re[btype==t].mean(),3) for t in np.unique(btype)})
print({t:(btype==t).sum() for t in np.unique(btype)})
```