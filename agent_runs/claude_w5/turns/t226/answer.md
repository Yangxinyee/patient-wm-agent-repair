H's 'target' field is the observed value, not the bin index, so the NLL indexing failed. The fine bin is 1 plus the number of bin edges below the value; the kernel's fbv(code, x) should compute that, and H also has an 'obs' field. Next I check both on a sample, then rerun the NLL comparison and the response regressions with the correct bin targets. Edited and original states, views and draws are unchanged, so the comparison stays matched.

```python
print('obs sample', [H[i]['obs'] for i in range(3)], '| fbv', [fbv(H[i]['code'], H[i]['target']) for i in range(3)])
def fine_bin(i):
    b = fbv(H[i]['code'], H[i]['target'])
    return int(b) if not hasattr(b, 'item') else int(b.item())
tgt_ = torch.tensor([fine_bin(i) for i in IDX])
print('bin range', int(tgt_.min()), int(tgt_.max()))
n0, p0 = nll_of(VO); n1, p1 = nll_of(VEd)
print('rows', len(IDX), 'NLL orig', n0.mean().round(4), 'edited', n1.mean().round(4), 'diff', (n1-n0).mean().round(5))
for c in np.unique(codes_):
    m = codes_ == c; print('  ', c, m.sum(), 'diff', (n1[m]-n0[m]).mean().round(5))
mean0 = (p0[:,1:]*cen[IDX]).sum(1)/p0[:,1:].sum(1); mean1 = (p1[:,1:]*cen[IDX]).sum(1)/p1[:,1:].sum(1)
dm = (mean1-mean0).numpy()
xs_I, xs_N, dg, dmp = [], [], [], []
pos = {i: j for j, i in enumerate(IDX)}
for r in DB:
    d = steps_all[r['sid']][r['k']]; R = {}; mystep(d, R=R); n3 = R['n3'][0].numpy()
    xI, xN = n3@wxI + bxI, n3@wxN + bxN
    for i in Hidx.get((r['sid'], r['k']+1), []):
        if H[i]['code'] == 'glucose_finger': xs_I.append((xI, xN)); dg.append(dm[pos[i]])
        if H[i]['code'] == 'map_arterial': xs_N.append((xI, xN)); dmp.append(dm[pos[i]])
A = np.c_[np.array(xs_I), np.ones(len(xs_I))]; cg = np.linalg.lstsq(A, np.array(dg), rcond=None)[0]
A2 = np.c_[np.array(xs_N), np.ones(len(xs_N))]; cm = np.linalg.lstsq(A2, np.array(dmp), rcond=None)[0]
print('next glucose shift ~ xI, xN, 1:', cg.round(3), '| next MAP shift ~ xI, xN, 1:', cm.round(3))
```