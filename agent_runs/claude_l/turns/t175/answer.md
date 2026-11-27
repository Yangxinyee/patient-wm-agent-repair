The run failed on an indexing bug, not a model issue. The deterministic view has shape (1, 192), so the softmax output is (1, 17) and `p[1:]` sliced the wrong axis. I'll take row 0 and repeat the comparison against the exact targets.

```python
for typ in [1,2,3]:
    for ref in ['cur','prev']:
        errs=[]
        for (s,k,c,pm,nl) in tests5:
            rf=reg_feats2(BUN[s],k,c,ref)
            fv=[rf['has'], math.log1p(rf['age'])/10, rf['z'],0,0,0,0,0]
            q=lnorm(W0, VD[(s,k)]+W0['type_emb.weight'][typ], 'query_norm')
            wc=q+W0['embed.code.weight'][c]+lin(W0,torch.tensor(fv,dtype=T),'last_numeric')+W0['embed.bins.value.weight'][rf['vb']]+W0['embed.bins.rate.weight'][0]
            lg=mlp(W0,wc,'attr_heads.value_bin')+W0['embed_transition.value_bin.weight'][c*17+rf['vb']]
            p=torch.softmax(lg,-1).numpy()[0]; errs.append((float((p[1:]*CENd[c]).sum())-pm, rf['has']))
        e_=np.array(errs); print(typ, ref, 'all med', round(float(np.median(np.abs(e_[:,0]))),4), 'has0 med', round(float(np.median(np.abs(e_[e_[:,1]==0,0]))),4), 'has1 med', round(float(np.median(np.abs(e_[e_[:,1]==1,0]))),4), 'n', len(e_))
```