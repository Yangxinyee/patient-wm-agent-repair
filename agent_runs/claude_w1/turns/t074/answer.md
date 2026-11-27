That was the missing piece. With empty sets encoded through the encoder in both directions, GELU activations, and mg/dL as unit row 3 and mmHg as row 2, the reading-only error falls from 0.70 to 0.135. The measurement encoder and the context are now essentially right.

Drug bundles are still at 0.44, so the action side has one or more wrong details. I'll search over:
- which unit row (4 or 5) is the dose unit "mg" and which is "units";
- the two rate-unit rows;
- whether END atoms carry unit embeddings;
- whether END atoms are encoded with the actions at all.

Mixed bundles don't exist in this sample: readings and drugs never share a minute. So only the action-only types matter here.

```python
def zpost_batch4(umap,rumap,act='gelu',types=None,nst=8,end_attr=False,end_in_action=True):
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
            isact=lambda a: (a['ev'] in ACTION) and (end_in_action or a['kind']==0)
            e_=enc_set2([atom_e(a,umap,rumap,end_attr) for a in B[k] if not isact(a)],'encode',act,False,False,'size')
            u_=enc_set2([atom_e(a,umap,rumap,end_attr) for a in B[k] if isact(a)],'encode_action',act,False,False,'size')
            zp=M2(np.concatenate([ctx,e_,u_]),'z_post',act)[:32]
            errs.append(np.abs(zp-X_[j1,960:]).mean()/np.abs(X_[j1,960:]).mean())
    return round(float(np.mean(errs)),3)
res=[]
for du in [(4,5),(5,4)]:
    for ru in [(2,3),(3,2)]:
        for ea in [False,True]:
            for eia in [True,False]:
                um_={'mg/dL':3,'mmHg':2,'mg':du[0],'units':du[1]}; rm_={'mcg/kg/min':ru[0],'units/hour':ru[1]}
                res.append((zpost_batch4(um_,rm_,types={'A'},end_attr=ea,end_in_action=eia),du,ru,ea,eia))
res.sort(key=lambda r:r[0]); print(res[:5])
```