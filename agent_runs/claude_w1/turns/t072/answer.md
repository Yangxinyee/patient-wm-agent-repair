The best unit mapping still leaves 70% relative error, so the units aren't the problem. Something structural in the measurement encoder or the context is missing.

The atom embedding looks right: every trained row is accounted for. One plausible gap remains: the two encoder MLPs may be residual (x + MLP(x)) rather than plain MLPs. The architecture note wouldn't necessarily say so. This is my last cheap structural test of the encoder:
- A sharp drop for some residual combination would revive the core replica.
- Otherwise I'll switch to measuring the network's responses directly with probe edits through the forward tool.

```python
def enc_set2(vs,prefix,act,res1=False,res2=False,empty='zero'):
    if len(vs)==0:
        if empty=='zero': return np.zeros(192)
        pooled=L(np.array([0.0]),prefix+'.size')
    else:
        A=np.stack(vs); h1=M2(A,prefix+'.mlp1',act)+(A if res1 else 0)
        pooled=h1.sum(0)/np.sqrt(len(vs))+L(np.array([np.log1p(len(vs))]),prefix+'.size')
    m=M2(pooled,prefix+'.mlp2',act)+(pooled if res2 else 0)
    return LNf(m,prefix+'.norm')
def zpost_batch2(umap,act='gelu',res1=False,res2=False,types=None,nst=8,empty='zero'):
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
            e_=enc_set2([atom_e(a,umap,{}) for a in B[k] if a['ev'] not in ACTION],'encode',act,res1,res2,empty)
            u_=enc_set2([atom_e(a,umap,{'mcg/kg/min':2,'units/hour':3}) for a in B[k] if a['ev'] in ACTION],'encode_action',act,res1,res2,'zero')
            zp=M2(np.concatenate([ctx,e_,u_]),'z_post',act)[:32]
            errs.append(np.abs(zp-X_[j1,960:]).mean()/np.abs(X_[j1,960:]).mean())
    return round(float(np.mean(errs)),3)
um={'mg/dL':4,'mmHg':2}
for r1 in [False,True]:
    for r2 in [False,True]:
        for act in ['gelu','relu']:
            print(r1,r2,act,'M:',zpost_batch2(um,act,r1,r2,types={'M'}),'A:',zpost_batch2(um,act,r1,r2,types={'A'}),'A-emptyE:',zpost_batch2(um,act,r1,r2,types={'A'},empty='size'))
```