Using the saved previous state, the second bundle's z posterior comes close (error 0.14 against a scale of 0.96, best with mg/dL→2 and mmHg→5) but is not exact. The remaining small differences may come from the `kq` term, the bin convention (< or ≤ the edge), or the gap scaling. Next I'll scan those three choices together on the second and third bundles. One combination giving near-zero error on both would pin down the encoder and z_post path.

```python
def atom_vec2(a_,umap,kq=True,le=False):
    c_=meta[a_['code']]; v=Wt['embed.code.weight'][c_['id']]+Wt['embed.kind.weight'][a_['kind']]+Wt['embed.ekind.weight'][ekid[c_['event_kind']]]
    if kq: v=v+Wt['embed.kq.weight'][0]
    nv=[0.]*6
    if 'value' in a_:
        e0=NUM[(a_['code'],'value')]; ed=np.array(e0['fine_bin_edges'])
        b_=1+int(np.sum(ed<=a_['value'])) if le else 1+int(np.sum(ed<a_['value']))
        nv[0]=zsc(a_['code'],'value',a_['value']); nv[1]=1.; v=v+Wt['embed.bins.value.weight'][min(b_,16)]
    if c_['unit'] is not None: v=v+Wt['embed.categorical.unit.weight'][umap[c_['unit']]]
    return v+Wt['embed.numeric.weight']@torch.tensor(nv)+Wt['embed.numeric.bias']
def gapx(x): return mlp(torch.tensor([[x]],dtype=torch.float32),'gap')[0]
res_=[]
for kq_,le_,gm in itertools.product([True,False],[False,True],['logs','log1ps','logh']):
    tot=0
    for bi in [1,2]:
        prev=r0[bi-1]; dt=float(bundles_B[s0_][bi][0]-bundles_B[s0_][bi-1][0])*3600
        e_=enc(torch.stack([atom_vec2(a_,{'mg/dL':2,'mmHg':5},kq_,le_) for a_ in bundles_B[s0_][bi][1] if a_['code'] not in ('insulin','norepinephrine')]),'encode')
        xg={'logs':np.log(dt)/10,'log1ps':np.log1p(dt)/10,'logh':np.log(dt/3600)/10}[gm]
        out_=X0[prev,:192]; h_=X0[prev,192:960].reshape(4,192)*torch.exp(-dt*flow)
        ctx=out_+lnp(h_.mean(0),'core.out_norm')+gapx(xg)
        zp=mlp(torch.cat([ctx,e_,torch.zeros(192)]),'z_post')[:32]
        tot=max(tot,float((zp-X0[r0[bi],960:992]).abs().max()))
    res_.append((round(tot,4),kq_,le_,gm))
res_.sort(); print(res_[:5])
```