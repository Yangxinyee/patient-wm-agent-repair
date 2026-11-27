Removing the unit term cuts the error from 0.14 to 0.094. So the unit row is either not used for readings or mapped wrongly; per the README, units may belong to the drug records. The error is still not zero, though.

Next I'll search jointly over the unit rows for mg/dL and mmHg (including none), the gap term (on or off) and the bins (on or off), on the same 45 bundles. I'll also time it, to see whether one combination gets near zero.

```python
import time; t0_=time.time()
caseinfo=[(p_,c_,dt,sidx,bi) for (p_,c_,dt,_),(sidx,bi) in zip(cases,[(s__,b__) for s__ in list(bundles_B.keys())[:3] for b__ in range(1,30) if all(a_['code'] not in ('insulin','norepinephrine','icu_admission') for a_ in bundles_B[s__][b__][1])])]
def av4(a_,ug,um,bins_):
    c_=meta[a_['code']]; v=Wt['embed.code.weight'][c_['id']]+Wt['embed.kind.weight'][a_['kind']]+Wt['embed.ekind.weight'][ekid[c_['event_kind']]]+Wt['embed.kq.weight'][0]+Wt['embed.numeric.bias']
    nv=[0.]*6
    if 'value' in a_:
        nv[0]=zsc(a_['code'],'value',a_['value']); nv[1]=1.
        if bins_: v=v+Wt['embed.bins.value.weight'][fbin(a_['code'],'value',a_['value'])]
    u_=ug if c_['unit']=='mg/dL' else um
    if u_ is not None: v=v+Wt['embed.categorical.unit.weight'][u_]
    return v+Wt['embed.numeric.weight']@torch.tensor(nv)
res_=[]
for ug in [None,1,2,3,4,5]:
    for um in [None,1,2,3,4,5]:
        for gp in [0,1]:
            E_=[]
            for p_,c_,dt,sidx,bi in caseinfo:
                out_=X0[p_,:192]; hf=X0[p_,192:960].reshape(4,192)*torch.exp(-dt*flow)
                e_=enc(torch.stack([av4(a_,ug,um,True) for a_ in bundles_B[sidx][bi][1]]),'encode')
                ctx=out_+lnp(hf.mean(0),'core.out_norm')+(gapx(np.log(dt)/10) if gp else 0)
                E_.append((mlp(torch.cat([ctx,e_,torch.zeros(192)]),'z_post')[:32]-X0[c_,960:992]).numpy())
            res_.append((round(float(np.sqrt((np.array(E_)**2).mean())),4),ug,um,gp))
res_.sort(key=lambda r:r[0]); print(res_[:6], round(time.time()-t0_,1),'s')
```