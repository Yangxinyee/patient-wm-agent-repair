First real progress: for drug-only bundles, encoding the empty set of measurements as LN(mlp2(size(0))) roughly halves the error, from 0.82 to 0.44. So empty sets go through the encoder rather than becoming zero vectors. The architecture note's "zero if the bundle has none" may be wrong, or may hold only in some places.

The obvious next test is the same rule for actions: reading-only bundles would then get u_k = encoder of the empty action set. If the reading-only error drops similarly, both encoders and the context are close. Whatever remains then points to a smaller detail.

```python
def zpost_batch3(umap,act='gelu',types=None,nst=8,emptyE='size',emptyU='size',res2=False):
    errs=[]
    for s_ in list(np.unique(sid_))[:nst]:
        ii=starts[s_]; B=bund[s_]
        for k in range(1,len(ii)):
            hasm=any(a['ev'] not in ACTION and a['kind']==0 for a in B[k]); hasa=any(a['ev'] in ACTION for a in B[k])
            t_=('M' if hasm else '')+('A' if hasa else '')
            if types and t_ not in types: continue
            j0,j1=ii[k-1],ii[k]; dtk=(hr_[j1]-hr_[j0])*3600
            hprev=X_[j0,192:960].reshape(4,192)*np.exp(-dtk*np.exp(Wt('core.flow.a')))[None,:]
            ctx=X_[j0,:192]+LNf(hprev.mean(0),'core.out_norm')+M2(np.array([np.log(dtk)/10]),'gap',act)
            e_=enc_set2([atom_e(a,umap,{}) for a in B[k] if a['ev'] not in ACTION],'encode',act,False,res2,emptyE)
            u_=enc_set2([atom_e(a,umap,{'mcg/kg/min':2,'units/hour':3}) for a in B[k] if a['ev'] in ACTION],'encode_action',act,False,res2,emptyU)
            zp=M2(np.concatenate([ctx,e_,u_]),'z_post',act)[:32]
            errs.append(np.abs(zp-X_[j1,960:]).mean()/np.abs(X_[j1,960:]).mean())
    return round(float(np.mean(errs)),3)
for act in ['gelu','relu']:
    for um_ in [{'mg/dL':4,'mmHg':2},{'mg/dL':2,'mmHg':3},{'mg/dL':3,'mmHg':2},{'mg/dL':5,'mmHg':2},{'mg/dL':2,'mmHg':5}]:
        print(act,um_,'M:',zpost_batch3(um_,act,types={'M'}),'A:',zpost_batch3(um_,act,types={'A'}),'MA:',zpost_batch3(um_,act,types={'MA'}))
```