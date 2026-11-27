A single P-space write direction doesn't work. It reproduces the bundle-specific belief directions poorly (cosine 0.67–0.78, residual 50–66%). Worse, it needs |δP| ≈ 2,000 per mg/dL of belief, larger than P itself (about 1,136). The prelude output, after its LayerNorm, is a weak lever on h_new; readings mostly reach the state through cross-attention to memory instead.

Better lever: b_in, which maps E linearly into h_new with no normalization in between. Revised design:
- The gated prelude-FFN units write a tiny "carrier" into P.
- It lands in E along a direction q that E naturally almost never uses.
- A rank-1 edit of b_in reads q·E and writes the belief direction into h_new with whatever gain is needed.

This only works if E has directions with near-zero natural spread, the carrier can reach them, and they aren't the LayerNorm's trivially blocked direction. Next I compute the eigen-spectrum of E's uncentered second moment over all bundles, and check how well the smallest-eigenvalue directions can be reached.

```python
Eall, Nall = [], []
for s in sids_B:
    for k, d in enumerate(steps_all[s]):
        R = {}; mystep(d, R=R); Eall.append(R['E'].numpy())
        Pm = R['P']; Nall.append(((Pm - Pm.mean(1, keepdim=True)) / Pm.std(1, unbiased=False, keepdim=True)).numpy())
Eall = np.concatenate(Eall); Nall = np.concatenate(Nall)
gam_pn = W64['core.prelude_norm.weight'].numpy(); bet_pn = W64['core.prelude_norm.bias'].numpy()
S_E = Eall.T @ Eall / len(Eall)
evE, VE = np.linalg.eigh(S_E)
print('E rows', Eall.shape, 'mean elem var', round(float(Eall.var(0).mean()),4), 'gamma range', gam_pn.min().round(3), gam_pn.max().round(3), '|beta|', np.linalg.norm(bet_pn).round(3))
print('smallest eig', evE[:10].round(8))
print('largest eig', evE[-3:].round(2))
one = np.ones(192)
for i in range(8):
    q = VE[:, i]; qg = q * gam_pn; reach = np.linalg.norm(qg - qg.mean()) / np.linalg.norm(qg)
    cos1 = abs(q @ (one / gam_pn)) / np.linalg.norm(one / gam_pn)
    print(i, 'eig', f"{evE[i]:.2e}", 'sd(qE)', f"{np.sqrt(np.var(Eall @ q)):.2e}", 'mean(qE)', f"{np.mean(Eall @ q):.2e}", 'reach', round(reach,3), 'cos(1/gamma)', round(cos1,3))
```