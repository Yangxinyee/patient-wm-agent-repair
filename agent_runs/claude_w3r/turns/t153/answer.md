`step_parts` returns (h-like state with norm about 855, E with norm 13.7 and std about 0.99 as a LayerNorm output should have, y, out, prelude attention weights). y and out match `step_at` exactly. Next I'll rerun the probe loop with `step_parts`, collecting E at every bundle that starts an infusion. Then I'll fit a grouped-CV ridge of each drug's rate on E. R² ≥ 0.95 would support a rank-1 `b_in` injection channel; much lower would refute it.

```python
t0 = time.time(); recE = []
for s in sorted(recs.stay_id.unique()):
    B0, rows0, EU0 = stay_data(int(s))
    for k in range(1, len(B0)):
        at = B0[k][1]
        ri = [a['rate'] for a in at if a['code']==ci and a['kind']==0]
        rn = [a['rate'] for a in at if a['code']==cn and a['kind']==0]
        if not ri and not rn: continue
        h0_, E_, y_, o_, att_ = step_parts(k)
        recE.append(dict(stay=int(s), k=k, t=B0[k][0], ri=(ri[0] if ri else 0.0), rn=(rn[0] if rn else 0.0),
                         has_i=bool(ri), has_n=bool(rn), E=E_.double().numpy(), o=o_.double().numpy(), y=y_.double().numpy()))
print(len(recE), 'bundles', round(time.time()-t0,1), 's')
EA = np.stack([r['E'].mean(0) for r in recE]); gr = np.array([r['stay'] for r in recE])
for nm, hk in [('ri','has_i'),('rn','has_n')]:
    yv = np.array([r[nm] for r in recE]); msk = np.array([r[hk] for r in recE])
    p = cvp(RidgeCV(alphas=np.logspace(-4,2,13)), EA[msk], yv[msk], groups=gr[msk], cv=GroupKFold(5))
    print(nm, 'n', msk.sum(), 'R2', round(1-np.var(yv[msk]-p)/np.var(yv[msk]),4), 'resid sd', round(np.std(yv[msk]-p),4), 'y sd', round(np.std(yv[msk]),3))
```