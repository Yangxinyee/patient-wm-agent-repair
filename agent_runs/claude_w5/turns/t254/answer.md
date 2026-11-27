`make_bundles` reproduces `bundles_B` exactly on a train stay, so the tuning bundles are built correctly and input construction is ruled out.

Next I'll measure the same per-bundle local-vs-tool error for the original model on train stays, using the fwd_2 states, split by drug versus reading bundles.
- If train drug bundles also show about 1e-3 error, the tool's float32 forward pass is ill-conditioned at drug bundles. Exact local agreement is impossible there, and my edit's decoders, which subtract large components, are too fragile.
- If train drug bundles match at about 1e-7 while tuning ones don't, something specific to these tuning stays is the cause.

```python
Z2 = np.load('/work/fwd_2/states_original.npz'); X2, S2 = Z2['x'], Z2['stay_id']
recs2=[]
for sid in list(np.unique(S2))[:6]:
    B = bundles_B[int(sid)]; idx = np.where(S2==sid)[0]
    RO = run_tf(B, step=stepO); LO = np.stack([st_vec(r) for r in RO])
    for j,i in enumerate(idx):
        isd = any(a['code'] in ('insulin','norepinephrine') and a['kind']==0 for a in B[j][1])
        d = X2[i]-LO[j]
        recs2.append(dict(sid=int(sid), j=j, drug=isd, err=np.linalg.norm(d)/np.linalg.norm(LO[j]),
                          err_out=np.linalg.norm(d[:192])/np.linalg.norm(LO[j][:192]), err_z=np.linalg.norm(d[960:])/np.linalg.norm(LO[j][960:]), nrm=np.linalg.norm(LO[j])))
C2 = pd.DataFrame(recs2)
print(C2.groupby('drug')[['err','err_out','err_z','nrm']].describe().T.round(6).to_string())
```