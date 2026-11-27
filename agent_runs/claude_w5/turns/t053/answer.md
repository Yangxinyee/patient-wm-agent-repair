No variant matches. The error stays at 3.9 or more against a scale of 7.4, and adding the "absent" bin and unit rows changes nothing, so those rows are probably zero.

The error falls as the first time gap shrinks, so next I'll scan the gap-encoder input directly over a wide range. If some value gives near-zero error, I've found the first-bundle convention. If the error never gets close to zero, the problem lies elsewhere: the start state, the token order, or the z_post input layout.

```python
print('bins0 norm', float(Wt['embed.bins.value.weight'][0].norm()), 'unit0', float(Wt['embed.categorical.unit.weight'][0].norm()))
e_=enc(emb_visit(11,3,0,0)[None],'encode'); u_=torch.zeros(192)
best_=[]
for xg in np.linspace(-4,2,61):
    g_=mlp(torch.tensor([[xg]],dtype=torch.float32),'gap')[0]
    for init in ['zero','lnzero']:
        out_=torch.zeros(192)
        ctx=out_+(lnp(torch.zeros(192),'core.out_norm') if init=='lnzero' else 0)+g_
        zp=mlp(torch.cat([ctx,e_,u_]),'z_post')[:32]
        best_.append((float((zp-zt).abs().max()),xg,init))
best_.sort(); print(best_[:5])
```