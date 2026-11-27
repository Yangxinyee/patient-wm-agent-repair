The state layout is consistent: the stored out equals out_norm(mean of the 4 slots) exactly. So the stored h is what the next step decays, and the posterior mismatch comes from my context formula, my encoders, or both.

There is a clean test bed for the context alone. Each stay's final bundle (25:10) holds only the two admission END atoms, with no values, units or bins, and has a known previous state. Last session's first-bundle test matched at about 0.976 using only a visit atom, which suggests visit-atom embeddings are roughly right.

Next, I will compute e for these 100 bundles with u = 0 and scan the context variants (flow units, gap scaling). If one variant gives correlation near 1, the context is settled and the remaining error lies in how measurement and drug atoms carry values. If none does, the problem is in the posterior network's input or my set encoder.

```python
lastb = np.where(np.r_[S3[1:]!=S3[:-1], True])[0]
print("last bundle hours", np.unique(np.round(H3[lastb],4))[:5], "atoms", [(a['code'],a['kind']) for a in atoms_by_bundle[mkey(S3[lastb[0]],H3[lastb[0]])]])
jl = np.searchsorted(idxb, lastb)  # positions within idxb
assert (idxb[jl]==lastb).all()
El = np.array([bundle_eu(atoms_by_bundle[mkey(S3[i],H3[i])], cfg0)[0] for i in lastb])
tl = X3[lastb,960:992]
for fl_nm, fl_c in [('sec',1.0),('min',1/60),('hr',1/3600),('none',0.0)]:
    Hf = Hraw_[jl]*np.exp(-(dtb[jl]*fl_c)[:,None]*np.exp(P['fa'])[None,:])[:,None,:]
    base_ = X3[pkb[jl],:192] + lnorm(Hf.mean(1), P['ow'], P['ob'])
    for g_nm, gin in {'ls10': np.log(dtb[jl])/10, 'lh10': np.log(dtb[jl]/3600)/10, 'ls': np.log(dtb[jl]), 'none': None}.items():
        cx = base_ + (0 if gin is None else mlp2(gin[:,None], 'gap'))
        zo_ = mlp2(np.c_[cx, El, np.zeros_like(El)], 'z_post')[:, :32]
        print(fl_nm, g_nm, round(np.corrcoef(zo_.ravel(), tl.ravel())[0,1],4), "MAD", np.abs(zo_-tl).mean().round(4))
```