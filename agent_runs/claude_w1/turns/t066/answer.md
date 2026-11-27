The z posterior replica is badly wrong: relative error about 0.75 for every unit mapping, so the mapping isn't the main problem. I also found a bug. Pandas NaN counts as true, so measurement atoms never received a unit embedding; that is why swapping the mg/dL and mmHg rows changed nothing.

I'll fix the bug, then check the coarse structural choices:
- the order of the three inputs [ctx, e, u];
- which half of the output is the mean;
- the activation;
- whether the gap term belongs in the context at all.

If one combination drops the error sharply, the encoders are roughly right and only the wiring was off. If none does, the encoders themselves are wrong.

```python
def atom_e(a,umap,rumap,end_attr=False):
    ev=a['ev']; v=EMB['embed.code.weight'][CID[ev]]+EMB['embed.kind.weight'][a['kind']]+EMB['embed.ekind.weight'][EV2EK[ev]]+EMB['embed.kq.weight'][0]
    num=np.zeros(6)
    du=a['du'] if isinstance(a['du'],str) else None
    unit=du if du else VOCAB['codes'][CID[ev]].get('unit')
    if a['kind']==0 or end_attr:
        if unit in umap: v=v+EMB['embed.categorical.unit.weight'][umap[unit]]
        if isinstance(a['ru'],str) and a['ru'] in rumap: v=v+EMB['embed.categorical.rate_unit.weight'][rumap[a['ru']]]
    for col,attr,bn,i in [('value','value','embed.bins.value.weight',0),('dose','dose','embed.bins.dose.weight',2),('rate','rate','embed.bins.rate.weight',4)]:
        x=a[col]
        if x is not None and x==x:
            num[i]=zs_(ev,attr,x); num[i+1]=1; v=v+EMB[bn][fb_(ev,attr,x)]
    return v+EMB['embed.numeric.weight']@num+EMB['embed.numeric.bias']
umap=dict(zip(['mg/dL','mmHg','mg','units'],[2,3,4,5])); rumap={'mcg/kg/min':2,'units/hour':3}
# collect inputs for ~400 bundles
CI=[];EI=[];UI=[];ZT=[];GI=[]
for s_ in list(np.unique(sid_))[:12]:
    ii=starts[s_]; B=bund[s_]
    for k in range(1,len(ii)):
        j0,j1=ii[k-1],ii[k]; dtk=(hr_[j1]-hr_[j0])*3600
        hprev=X_[j0,192:960].reshape(4,192)*np.exp(-dtk*np.exp(Wt('core.flow.a')))[None,:]
        CI.append(X_[j0,:192]+LNf(hprev.mean(0),'core.out_norm')); GI.append(np.log(dtk)/10)
        EI.append(enc_set([atom_e(a,umap,rumap) for a in B[k] if a['ev'] not in ACTION],'encode','gelu'))
        UI.append(enc_set([atom_e(a,umap,rumap) for a in B[k] if a['ev'] in ACTION],'encode_action','gelu'))
        ZT.append(X_[j1,960:])
CI,EI,UI,ZT,GI=map(np.array,(CI,EI,UI,ZT,GI))
print('n',len(ZT))
res=[]
for act in ['gelu','relu']:
    gp=M2(GI[:,None],'gap',act)
    for withgap in [True,False]:
        C_=CI+gp if withgap else CI
        for order in itertools.permutations(['c','e','u']):
            inp=np.concatenate([{'c':C_,'e':EI,'u':UI}[o] for o in order],1)
            o_=M2(inp,'z_post',act)
            for half in [0,1]:
                zp=o_[:,32*half:32*half+32]
                res.append((round(np.abs(zp-ZT).mean()/np.abs(ZT).mean(),3),act,withgap,order,half))
res.sort(key=lambda r:r[0]); print(res[:5])
```