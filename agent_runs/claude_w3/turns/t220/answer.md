The older helpers are `atom_vec(code, kind, ekind, value, rate, dose, P0, opt)`, `set_enc_(vectors, prefix, P0)`, `mlp_` and `ln_`. Next I will recompute the z-posterior check with these helpers, split by bundle type. If drug bundles come out near 0.1 or better, I can trust the local u_k.

```python
def enc_old(df, opt={}):
    mv=[];av=[]
    for r in df.itertuples():
        v=atom_vec(int(r.code),int(r.kind),int(r.ekind),r.value,r.rate,r.dose,P0,opt)
        (av if int(r.code) in (8,9) else mv).append(v)
    e=set_enc_(mv,'encode',P0) if mv else torch.zeros(192,dtype=torch.float64)
    u=set_enc_(av,'encode_action',P0) if av else torch.zeros(192,dtype=torch.float64)
    return e.float(),u.float()
res=[]
for s in np.unique(sid4):
    idx=np.where(sid4==s)[0]; B=stay_bundles(s)
    for a in range(1,len(idx)):
        i,ip=idx[a],idx[a-1]; dt=(B[a].minute.iloc[0]-B[a-1].minute.iloc[0])*60.
        e,u=enc_old(B[a]); c=ctx_of(X4[ip,:192],X4[ip,192:960],dt)
        zp=mlp(torch.cat([c,e,u]),'z_post')[:32]; zt=X4[i,960:]
        res.append((int(S4['has_insulin'][i])+2*int(S4['has_norepinephrine'][i]), float((zp-zt).norm()/zt.norm())))
r=np.array(res)
for t in range(4): print(t,(r[:,0]==t).sum(), np.median(r[r[:,0]==t,1]).round(3))
```