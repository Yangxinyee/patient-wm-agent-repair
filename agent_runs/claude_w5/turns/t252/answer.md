The tool's states for both models on 13 tuning stays have arrived (689 bundle rows). I'll compute my local original and edited trajectories for the same stays and compare per bundle:
1. Original vs tool, to confirm row alignment and an error of about 1e-6.
2. Edited vs tool, to find the first divergent bundle.
3. The edit's effect, edited minus original, local against tool: cosine and norm ratio at drug and non-drug bundles.

A cosine near +1 with ratio near 1 would mean the computation matches. A cosine near −1 would mean a sign flip. A large tool-only effect at non-drug bundles would point to the gate or the quiet directions.

```python
Z5o = np.load('/work/fwd_5/states_original.npz'); Z5e = np.load('/work/fwd_5/states_edited.npz')
Xo5, Xe5, S5 = Z5o['x'], Z5e['x'], Z5o['stay_id']
def st_vec(r): return np.concatenate([r[0].numpy().ravel(), r[1].numpy().ravel(), r[2].numpy().ravel()[:32]])
recs=[]
for sid in np.unique(S5):
    B = TB[int(sid)]; idx = np.where(S5==sid)[0]
    RO = run_tf(B, step=stepO); RE = run_tf(B, step=stepE)
    LO = np.stack([st_vec(r) for r in RO]); LE = np.stack([st_vec(r) for r in RE])
    assert len(idx)==len(LO), (sid, len(idx), len(LO))
    for j,(i) in enumerate(idx):
        to, te, lo, le = Xo5[i], Xe5[i], LO[j], LE[j]
        dtool, dloc = te-to, le-lo
        isd = any(a['code'] in ('insulin','norepinephrine') and a['kind']==0 for a in B[j][1])
        recs.append(dict(sid=int(sid), j=j, t=B[j][0], drug=isd,
            err_o=np.linalg.norm(to-lo)/np.linalg.norm(lo), err_e=np.linalg.norm(te-le)/np.linalg.norm(le),
            nd_tool=np.linalg.norm(dtool), nd_loc=np.linalg.norm(dloc),
            cos=float(dtool@dloc/(np.linalg.norm(dtool)*np.linalg.norm(dloc)+1e-30))))
CMP = pd.DataFrame(recs)
print(CMP[['err_o','err_e','nd_tool','nd_loc','cos']].describe().round(5).to_string())
print(CMP.groupby('drug')[['err_e','nd_tool','nd_loc','cos']].median().round(5))
s0 = CMP[CMP.sid==CMP.sid.iloc[0]]
print(s0.head(30).round(5).to_string())
```