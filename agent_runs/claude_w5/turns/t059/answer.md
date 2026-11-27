The order [ctx, e_k, u_k] and the `kq[0]` term both look right, but the first bundle still doesn't match. The start state may simply be different.

To set that question aside, I'll test z_post on the second bundle instead. It holds the first readings at hour −1, so I can take the previous `out` and h from the saved state row and apply the documented time flow. This needs measurement-atom embeddings, so I'll try each assignment of the two value units to the four trained unit rows. A near-zero error would confirm the encoder and z_post path, and show that the first-bundle problem is about the start state.

```python
meta={c_['name']:c_ for c_ in VOCAB['codes'] if 'event_kind' in c_}
ekid={'measurement':1,'drug_admin':2,'visit':3,'visit_detail':4}
NUM={(e_['code'],e_['attribute']):e_ for e_ in num}
def zsc(code,attr,x):
    e_=NUM[(code,attr)]; return float(np.clip((x-e_['median'])/e_['iqr'],-10,10))
def fbin(code,attr,x):
    e_=NUM[(code,attr)]; return 1+int(np.sum(np.array(e_['fine_bin_edges'])<x))
def atom_vec(a_,umap,rumap=None,kq=0):
    c_=meta[a_['code']]; v=Wt['embed.code.weight'][c_['id']]+Wt['embed.kind.weight'][a_['kind']]+Wt['embed.ekind.weight'][ekid[c_['event_kind']]]+Wt['embed.kq.weight'][kq]
    nv=[0.]*6
    if 'value' in a_:
        nv[0]=zsc(a_['code'],'value',a_['value']); nv[1]=1.; v=v+Wt['embed.bins.value.weight'][fbin(a_['code'],'value',a_['value'])]
    if c_['unit'] is not None and umap is not None: v=v+Wt['embed.categorical.unit.weight'][umap[c_['unit']]]
    return v+Wt['embed.numeric.weight']@torch.tensor(nv)+Wt['embed.numeric.bias']
flow=torch.exp(Wt['core.flow.a'])
def ctx_from(row,dt):
    out_=X0[row,:192]; h_=X0[row,192:960].reshape(4,192)*torch.exp(-dt*flow)
    return out_+lnp(h_.mean(0),'core.out_norm')+gapf(dt)
ats=bundles_B[s0_][1][1]; zt2=X0[r0[1],960:992]
res_=[]
for ug,um in itertools.permutations([2,3,4,5],2):
    umap={'mg/dL':ug,'mmHg':um}
    e_=enc(torch.stack([atom_vec(a_,umap) for a_ in ats]),'encode')
    zp=mlp(torch.cat([ctx_from(r0[0],3600.0),e_,torch.zeros(192)]),'z_post')[:32]
    res_.append((round(float((zp-zt2).abs().max()),4),ug,um))
res_.sort(); print(res_[:4], 'scale', round(float(zt2.abs().max()),2))
```