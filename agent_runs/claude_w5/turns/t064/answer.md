With mg/dL→4 and mmHg→5 the errors are more even but still 0.15–0.35. Next I'll test whether the leftover error is the same vector in every bundle; if so, a constant input term is missing. I'll print the cosine similarity of the error vectors and their mean size relative to the target scale. High cosine would mean a missing constant (for example in ctx); low cosine would point to something that varies with the input.

```python
U2={'mg/dL':4,'mmHg':5}
def zpred(sidx,bi,kq_=True):
    rr_=np.where(SID==sidx)[0]; bl_=bundles_B[sidx]
    prev=rr_[bi-1]; dt=float(bl_[bi][0]-bl_[bi-1][0])*3600
    e_=enc(torch.stack([atom_vec2(a_,U2,kq_) for a_ in bl_[bi][1]]),'encode')
    out_=X0[prev,:192]; h_=X0[prev,192:960].reshape(4,192)*torch.exp(-dt*flow)
    zp=mlp(torch.cat([out_+lnp(h_.mean(0),'core.out_norm')+gapx(np.log(dt)/10),e_,torch.zeros(192)]),'z_post')[:32]
    return zp, X0[rr_[bi],960:992]
errs_=[]
for sidx in list(bundles_B.keys())[:3]:
    for bi in range(1,30):
        if all(a_['code'] not in ('insulin','norepinephrine','icu_admission') for a_ in bundles_B[sidx][bi][1]):
            zp,zt_=zpred(sidx,bi); errs_.append((zp-zt_).numpy())
E_=np.array(errs_); cs=(E_@E_.T)/np.outer(np.linalg.norm(E_,axis=1),np.linalg.norm(E_,axis=1))
print(len(E_), 'mean cos', round(float(cs[np.triu_indices(len(E_),1)].mean()),3), 'err rms', round(float(np.sqrt((E_**2).mean())),3))
```