The code table gives each code's event kind and unit. Next I'll start rebuilding the forward pass locally, beginning with the first bundle of a stay (the hospital admission event, which has no attributes). The z posterior mean after that bundle is saved in the states file, so I can check my embedding, set encoder, context and z_post computation against it directly. I'll try a few variants of the unknown details (start state, first time gap, which "absent" embeddings are added) and print the error for each. An error near zero for one variant would confirm those details.

```python
import torch, torch.nn.functional as Fn, itertools
Wt={n_:torch.tensor(load_original(n_)) for n_ in NAMES}
def lin(x,p,bias=True): 
    y=x@Wt[p+'.weight'].T
    return y+Wt[p+'.bias'] if bias and (p+'.bias') in Wt else y
def mlp(x,p): return lin(Fn.gelu(lin(x,p+'.0')),p+'.2')
def lnp(x,p): return Fn.layer_norm(x,(192,),Wt[p+'.weight'],Wt[p+'.bias'],1e-5)
def gapf(dt): return mlp(torch.tensor([[np.log(max(dt,1.0))/10]],dtype=torch.float32),'gap')[0]
X0=torch.tensor(Z['x']); SID=Z['stay_id']
s0_=SID[0]; r0=np.where(SID==s0_)[0]
atom0=bundles_B[s0_][0][1][0]; print(atom0)
def emb_visit(code_id, ek, add_bins0, add_cat0):
    v=Wt['embed.code.weight'][code_id]+Wt['embed.kind.weight'][0]+Wt['embed.ekind.weight'][ek]+Wt['embed.numeric.bias']
    if add_bins0: v=v+Wt['embed.bins.value.weight'][0]+Wt['embed.bins.dose.weight'][0]+Wt['embed.bins.rate.weight'][0]
    if add_cat0: v=v+Wt['embed.categorical.unit.weight'][0]
    return v
def enc(atoms_v,p):
    n_=atoms_v.shape[0]
    s_=mlp(atoms_v,p+'.mlp1').sum(0)/np.sqrt(n_)+lin(torch.tensor([np.log(1+n_)],dtype=torch.float32),p+'.size')
    return lnp(mlp(s_,p+'.mlp2'),p+'.norm')
zt=X0[r0[0],960:992]
for ab,ac,dt0 in itertools.product([0,1],[0,1],[1.0,3600.0,7200.0]):
    e_=enc(emb_visit(11,3,ab,ac)[None],'encode'); u_=torch.zeros(192)
    out_=torch.zeros(192); hm_=torch.zeros(192)
    ctx=out_+lnp(hm_,'core.out_norm')+gapf(dt0)
    zp=mlp(torch.cat([ctx,e_,u_]),'z_post')[:32]
    print(ab,ac,dt0,'err %.4f'%float((zp-zt).abs().max()), 'scale %.3f'%float(zt.abs().max()))
```