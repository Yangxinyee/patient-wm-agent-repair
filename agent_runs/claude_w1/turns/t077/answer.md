Bin-edge handling isn't the cause: counting edges strictly below the value stays better (0.158) than counting edges at or below (0.195). The drug-bundle error rose only because I forgot to pass the END-atom setting in this call, a slip on my part and not a new finding.

To locate the measurement problem, I'll split the reading-only error by bundle composition: which codes appear, and whether a visit atom is present. If one code, say arterial MAP or lab glucose, carries most of the error, its atoms are being embedded differently.

```python
atom_e=atom_e_side('left')
def zpost_rows(types={'M'},nst=30):
    out=[]
    for s_ in list(np.unique(sid_))[:nst]:
        ii=starts[s_]; B=bund[s_]
        for k in range(1,len(ii)):
            hasm=any(a['ev'] not in ACTION and a['kind']==0 for a in B[k]); hasa=any(a['ev'] in ACTION for a in B[k])
            t_=('M' if hasm else '')+('A' if hasa else '')
            if t_ not in types: continue
            j0,j1=ii[k-1],ii[k]; dtk=(hr_[j1]-hr_[j0])*3600
            hprev=X_[j0,192:960].reshape(4,192)*np.exp(-dtk*np.exp(Wt('core.flow.a')))[None,:]
            ctx=X_[j0,:192]+LNf(hprev.mean(0),'core.out_norm')+M2(np.array([np.log(dtk)/10]),'gap','gelu')
            isact=lambda a: (a['ev'] in ACTION) and a['kind']==0
            e_=enc_set2([atom_e(a,UM,RM) for a in B[k] if not isact(a)],'encode','gelu',False,False,'size')
            u_=enc_set2([atom_e(a,UM,RM) for a in B[k] if isact(a)],'encode_action','gelu',False,False,'size')
            zp=M2(np.concatenate([ctx,e_,u_]),'z_post','gelu')[:32]
            comp='+'.join(sorted(set(a['ev']+('_END' if a['kind']==2 else '') for a in B[k])))
            out.append((comp, np.abs(zp-X_[j1,960:]).mean()/np.abs(X_[j1,960:]).mean(), hr_[j1]))
    return pd.DataFrame(out,columns=['comp','err','hour'])
zr=zpost_rows()
print(zr.groupby('comp').err.agg(['mean','size']).sort_values('size',ascending=False).round(3).head(12).to_string())
print(zr.groupby(pd.cut(zr.hour,[-3,0,2,6,12,26])).err.mean().round(3).to_dict())
```