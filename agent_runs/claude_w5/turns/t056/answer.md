The correlation is 0.94, so the structure is nearly right but one term is missing. The embedding tables show which rows were trained. Rows with norm about 0.28 look untouched since initialisation. `embed.kq` row 0 is trained (0.66), so it is probably added to every atom.

Next I'll add `embed.kq[0]` to the atom vector and rescan the gap input. Near-zero error would confirm it.

```python
def emb_visit2(code_id, ek, kq=0):
    return Wt['embed.code.weight'][code_id]+Wt['embed.kind.weight'][0]+Wt['embed.ekind.weight'][ek]+Wt['embed.numeric.bias']+Wt['embed.kq.weight'][kq]
e_=enc(emb_visit2(11,3)[None],'encode')
res_=[]
for xg in np.linspace(-3,1,81):
    g_=mlp(torch.tensor([[xg]],dtype=torch.float32),'gap')[0]
    for init in [0,1]:
        ctx=g_+(lnp(torch.zeros(192),'core.out_norm') if init else 0)
        zp=mlp(torch.cat([ctx,e_,torch.zeros(192)]),'z_post')[:32]
        res_.append((round(float((zp-zt).abs().max()),4),round(xg,3),init))
res_.sort(); print(res_[:4])
```