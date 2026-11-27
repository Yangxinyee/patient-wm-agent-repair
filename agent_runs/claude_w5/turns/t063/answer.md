The large errors follow the arterial MAP value. Glucose-only bundles fit well (0.13 and 0.24), while bundles with low MAP fit worst (MAP 58 gives 0.96, MAP 60 gives 0.61). So the MAP atom embedding is probably wrong; the unit row for mmHg was chosen from a single bundle.

Next I'll redo the search over all unit assignments jointly on the six reading-only bundles, crossed with the bin convention and `kq`. One configuration giving near-zero error on all of them would confirm it.

```python
def zerr2(bi,umap,kq_,le_):
    prev=r0[bi-1]; dt=float(bundles_B[s0_][bi][0]-bundles_B[s0_][bi-1][0])*3600
    ms=bundles_B[s0_][bi][1]
    e_=enc(torch.stack([atom_vec2(a_,umap,kq_,le_) for a_ in ms]),'encode')
    out_=X0[prev,:192]; h_=X0[prev,192:960].reshape(4,192)*torch.exp(-dt*flow)
    zp=mlp(torch.cat([out_+lnp(h_.mean(0),'core.out_norm')+gapx(np.log(dt)/10),e_,torch.zeros(192)]),'z_post')[:32]
    return float((zp-X0[r0[bi],960:992]).abs().max())
res_=[]
for ug,um in itertools.permutations([1,2,3,4,5],2):
    for kq_,le_ in itertools.product([True,False],[False,True]):
        errs=[zerr2(bi,{'mg/dL':ug,'mmHg':um},kq_,le_) for bi in [1,4,6,8,10,12]]
        res_.append((round(max(errs),3),ug,um,kq_,le_,[round(e,2) for e in errs]))
res_.sort(key=lambda r:r[0]); print(res_[:4])
```