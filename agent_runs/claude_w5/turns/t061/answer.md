Across bundles 2 and 3 the worst error is 1.78, so bundle 3 is far off. It holds the ICU admission event (a different event kind), which may be embedded differently.

Next I'll print the error bundle by bundle for the measurement-only bundles in the first stay, under four settings: with and without `kq`, crossed with gap measured in seconds or in hours. I want to see whether all ordinary reading bundles match with one setting, and only the special events differ.

```python
def zerr(bi,kq_,gm,umap={'mg/dL':2,'mmHg':5}):
    prev=r0[bi-1]; dt=float(bundles_B[s0_][bi][0]-bundles_B[s0_][bi-1][0])*3600
    ms=[a_ for a_ in bundles_B[s0_][bi][1] if a_['code'] not in ('insulin','norepinephrine')]
    if len(ms)==0 or len(ms)<len(bundles_B[s0_][bi][1]): return None
    e_=enc(torch.stack([atom_vec2(a_,umap,kq_) for a_ in ms]),'encode')
    xg=np.log(dt)/10 if gm=='logs' else np.log(dt/3600)/10
    out_=X0[prev,:192]; h_=X0[prev,192:960].reshape(4,192)*torch.exp(-dt*flow)
    zp=mlp(torch.cat([out_+lnp(h_.mean(0),'core.out_norm')+gapx(xg),e_,torch.zeros(192)]),'z_post')[:32]
    return round(float((zp-X0[r0[bi],960:992]).abs().max()),3)
for kq_,gm in itertools.product([True,False],['logs','logh']):
    print(kq_,gm,[(bi,zerr(bi,kq_,gm)) for bi in range(1,14) if zerr(bi,kq_,gm) is not None])
```