The run failed because an earlier loop variable reused the name `hi_` (the per-bundle insulin flag) for an integer. I'll rebuild the insulin and NE flags from the atoms of each bundle and rerun the same Jacobian experiment. The design question is unchanged: is there a z direction that consistently lowers the next glucose prediction while leaving MAP alone?

```python
hi_ = np.zeros(5156, bool); hn_ = np.zeros(5156, bool)
for key,atoms in atoms_by_bundle.items():
    i = rowkey[(int(key[0]), int(key[1]))]
    hi_[i] = any(a['code']=='insulin' and a.get('kind',0)==0 for a in atoms)
    hn_[i] = any(a['code']=='norepinephrine' and a.get('kind',0)==0 for a in atoms)
insb = [k for k in range(5155) if hi_[k] and S3[k+1]==S3[k]]
samp = np.random.default_rng(0).choice(insb, size=min(300,len(insb)), replace=False)
Vdef = {'incl':1,'unit':'s','age':'both'}
eps_ = 0.5; JG = []; t0_ = time.time()
for k in samp:
    s_ = int(S3[k]); ks_ = list(np.where(S3==s_)[0]); i0 = ks_.index(k)
    hp_ = torch.tensor(X3[k-1,192:960].reshape(4,192), dtype=WTd) if i0>0 else torch.zeros(4,192,dtype=WTd)
    op_ = torch.tensor(X3[k-1,:192], dtype=WTd) if i0>0 else torch.zeros(192,dtype=WTd)
    dtk = dts[k]; hf = hp_*torch.exp(-dtk*torch.exp(WT['core.flow.a'])); cx = ctx_from(op_, hf, dtk); z0 = zpost_mean(cx, Et[k], Ut[k])
    idx_m = np.arange(max(ks_[0], k+1-128), k+1); ages = (tmin[k]-tmin[idx_m])*60.0
    hn = hr_[k+1]; dtn = float(dts[k+1])
    rows = []
    for i in range(-1,32):
        zz_ = z0.clone()
        if i>=0: zz_[i] += eps_
        h1, o1 = core_step(hf, Et[k], Ut[k], zz_, Et[idx_m], ages, Vdef)
        rows.append(preds_after(s_, hn, o1, h1, dtn))
    rows = np.array(rows); JG.append((rows[1:]-rows[0])/eps_)
JG = np.array(JG)
print('time', round(time.time()-t0_,1), 'JG shape', JG.shape, 'n ins bundles', len(insb))
mJ = JG.mean(0); sJ = JG.std(0)
print('mean dGF/dz', mJ[:,0].round(2)); print('mean dMA/dz', mJ[:,2].round(2))
print('consistency |mean|/sd gf', (np.abs(mJ[:,0])/(sJ[:,0]+1e-9)).round(2))
```