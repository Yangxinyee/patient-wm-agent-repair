The relation `out = LN(mean slots of h)` holds exactly, so the state layout is right. But my z_post error (RMS 0.22) is about 40% of the signal (RMS 0.52), so ctx or e_k is still wrong.

Next I'll test versions of ctx (with or without the time flow on h, with or without `out`, with or without the gap term) on the same 45 reading-only bundles. If one gives a much lower RMS error, that piece was the problem; if none does, the encoder e_k is the culprit.

```python
cases=[]
for sidx in list(bundles_B.keys())[:3]:
    rr_=np.where(SID==sidx)[0]; bl_=bundles_B[sidx]
    for bi in range(1,30):
        if all(a_['code'] not in ('insulin','norepinephrine','icu_admission') for a_ in bl_[bi][1]):
            dt=float(bl_[bi][0]-bl_[bi-1][0])*3600
            cases.append((rr_[bi-1],rr_[bi],dt,enc(torch.stack([atom_vec2(a_,U2,True) for a_ in bl_[bi][1]]),'encode')))
def ctxv(prev,dt,v):
    out_=X0[prev,:192]; h0_=X0[prev,192:960].reshape(4,192); hf=h0_*torch.exp(-dt*flow)
    g_=gapx(np.log(dt)/10)
    return {'a':out_+lnp(hf.mean(0),'core.out_norm')+g_,'b':out_+lnp(h0_.mean(0),'core.out_norm')+g_,
            'c':lnp(hf.mean(0),'core.out_norm')+g_,'d':out_+g_,'e':out_+lnp(hf.mean(0),'core.out_norm'),'f':out_+hf.mean(0)+g_}[v]
for v in 'abcdef':
    E_=np.array([(mlp(torch.cat([ctxv(p_,dt,v),e_,torch.zeros(192)]),'z_post')[:32]-X0[c_,960:992]).numpy() for p_,c_,dt,e_ in cases])
    print(v, round(float(np.sqrt((E_**2).mean())),4))
```