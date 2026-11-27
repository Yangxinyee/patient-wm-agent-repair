That check failed: no 32-coordinate slice of [ctx, e_k, u_k] matches the probe's z means (best mean error 0.65), so fwd_4 gives no ground truth for measurement e_k.

Next I'll test a full CPU replica of the step at drug bundles anyway, using exact e for drug bundles and my approximate e for measurement bundles. Admission and END-only bundles are handled by whichever helper accepts them. If the replica's post-bundle state comes close to the GPU's (relative error of a few percent), the approximation of M is good enough to compute injection responses on CPU. First I build the e vector for every bundle row.

```python
# build e for all 5218 rows: exact for drug rows (from XIN3), approx otherwise
e_all = torch.zeros(len(st), 192); src = np.array(['']*len(st), dtype=object)
e_all[rows] = XIN3[:, 192:384].float(); src[rows] = 'drug'
grp_all = grp
nfail = 0
for i in np.where(~dr)[0]:
    g = grp_all.get((st[i], hr[i]))
    try:
        if g is not None and g.event.isin(mcodes).all():
            atoms = [meas_parts(r.event, float(r.value)) for r in g.itertuples()]
            e_all[i] = enc_e(atoms).reshape(-1); src[i] = 'meas'
        else:
            e_all[i] = e_drug_bundle(st[i], hr[i]).reshape(-1).float(); src[i] = 'other'
    except Exception as ex:
        nfail += 1; src[i] = 'fail'
print('fail', nfail, {k: int((src == k).sum()) for k in ['drug', 'meas', 'other', 'fail']})
```