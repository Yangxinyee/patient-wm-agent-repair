e_k for reading bundles still won't replicate. The best match is about 0.35, whatever I change in the value or table terms. I'll stop spending runs on it, because the edit does not need it.

My injection will act only at drug bundles, gated on u_k. At those bundles everything z_post sees is exactly reproducible:
- **ctx** from the states;
- **e_k**, which is either the empty-set encoding or the END-atom encoding, both exact;
- **u_k**, exact.

So I can compute the z_post input exactly wherever the edit acts. At reading bundles I only need the repurposed units to stay off, and I can guarantee that by gating on u_k equal to the empty-set value.

The open design question is which direction in z-space to push at insulin bundles. The natural choice is the direction the posterior already uses to encode "this glucose reading came in higher or lower than expected". Next I'll regress the original model's z posterior mean at finger-glucose bundles on the reading surprise, using fwd_3 states and fwd_2 predictions. If z carries glucose news clearly (a sizeable R²), its coefficient vector is the injection direction. If R² is near zero, glucose news reaches the state through e_k instead, and I'll need another route.

```python
S3 = np.load('/work/fwd_3/states_original.npz'); X3 = S3['x']; Z3 = X3[:, 960:992]
sid3, hr3 = S3['stay_id'], S3['hour']
F2 = pd.read_csv('/work/fwd_2/factual_original.csv'); print(F2.columns.tolist(), len(F2))
mcol = [c for c in F2.columns if 'mean' in c.lower()][0]
F2['hk'] = F2.hour.round(4)
k3 = pd.DataFrame({'stay_id': sid3, 'hk': np.round(hr3,4), 'row': np.arange(len(sid3))})
for evn in ['glucose_finger','map_arterial']:
    fm = F2[F2.event==evn].merge(k3, on=['stay_id','hk'], how='inner')
    s_ = (fm.value - fm[mcol]).values; Zr = Z3[fm.row.values]
    A_ = np.c_[np.ones(len(s_)), s_]; coef, *_ = np.linalg.lstsq(A_, Zr, rcond=None)
    pred_ = A_ @ coef; r2 = 1 - ((Zr-pred_)**2).sum(0)/((Zr-Zr.mean(0))**2).sum(0)
    # how well does z (all dims) predict the surprise?
    B_ = np.c_[np.ones(len(s_)), Zr]; cb, *_ = np.linalg.lstsq(B_, s_, rcond=None); r2s = 1 - ((s_-B_@cb)**2).sum()/((s_-s_.mean())**2).sum()
    print(evn, 'n', len(s_), 'surprise sd', s_.std().round(2), '| R2 z->surprise', round(r2s,3), '| top dims by R2', np.argsort(-r2)[:5], np.sort(r2)[::-1][:5].round(3))
```