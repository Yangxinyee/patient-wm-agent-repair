The refit shows the readouts work where it matters. The edited-minus-recorded readout difference, summed over drug bundles, is within 1% of the true dose change for insulin at every multiplier (0.995 to 1.006). For NE it is within 1.5% at ½, 2 and 3, but 7% low at ×0.

The large per-point errors have a different cause: they come from insulin rates above 11.5. The model's input encoding clips the z-score at 10, so it literally cannot tell 12 from 22.5. Those points should have their target capped at 11.5, not counted as readout failures; this only limits the ×3 insulin edit at high rates.

One more concern: the presence gates for NE and for each pattern have tiny margins in u-space (|w| about 350). That makes their pre-activations large and could make them sensitive to numerical precision on the GPU.

Next run:
- refit with capped targets;
- find which u's from opposite presence classes nearly coincide;
- build the clone part of the edit (the any-drug gate plus the 25 component units in z_post, and the matching z_in);
- check with the exact prelude copy that the model is otherwise unchanged: E should be unchanged at drug bundles, where the z token copies u, and at non-drug bundles, where the z token is a fixed token that gets no attention.

Confirmation: a maximum change in E that is tiny compared with E's variation across bundles, at both bundle types.

```python
import numpy as np, torch
from sklearn.linear_model import Ridge
capI = 11.5
ro3 = {}
for nmr, k, yname in [('I_only', 1, 'a'), ('N_only', 2, 'b'), ('both_I', 3, 'a'), ('both_N', 3, 'b')]:
    yG = aG if yname == 'a' else bG; yRv = aR if yname == 'a' else bR
    X = np.vstack([UG[pat == k], UR[patR == k]]); y = np.r_[yG[pat == k], yRv[patR == k]]
    if yname == 'a': y = np.minimum(y, capI)
    w = np.r_[np.full((pat == k).sum(), 0.05), np.ones((patR == k).sum())]
    ro3[nmr] = Ridge(alpha=1e-8).fit(X, y, sample_weight=w)
    yR_c = np.minimum(yRv[patR == k], capI) if yname == 'a' else yRv[patR == k]
    pR = ro3[nmr].predict(UR[patR == k])
    print(f'{nmr}: capped realistic rms {np.sqrt(((pR - yR_c)**2).mean()):.4f} max {np.abs(pR - yR_c).max():.4f}')
allU = np.vstack([UG, UR]); allA = np.r_[aG, aR]; allB = np.r_[bG, bR]
hasIall = ~np.isnan(allA); hasNall = ~np.isnan(allB)
for nmc, cls in [('ins', hasIall), ('ne', hasNall)]:
    P_ = allU[cls]; Q_ = allU[~cls]
    d2 = (P_**2).sum(1)[:, None] + (Q_**2).sum(1)[None, :] - 2 * P_ @ Q_.T
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    print(f'{nmc}: closest opposite pair dist {np.sqrt(max(d2[i, j], 0)):.4f}: present (I={allA[cls][i]}, N={allB[cls][i]}) vs absent (I={allA[~cls][j]}, N={allB[~cls][j]})')
Wo_ = {n: load_original(n) for n in ['z_in.weight', 'z_in.bias']}
zt_orig_n = TOKn[:, 6].float(); zt_orig_d = TOK[:, 6].float()
cand_T0 = {'bias': Wo_['z_in.bias'].float(), 'mean_meas': zt_orig_n.mean(0)}
E_ref_d = Pd['E']; E_ref_n = Pn['E']
sdE = float(E_ref_d.std(0).mean())
for nmT, T0 in cand_T0.items():
    Tn = TOKn.float().clone(); Tn[:, 6] = T0
    dEn = (prelude2(Tn, Wp)['E'] - E_ref_n).abs()
    print(f'T0={nmT}: non-drug E change max {float(dEn.max()):.4f} mean {float(dEn.mean()):.5f} (E sd across bundles {sdE:.3f})')
uk_t = torch.tensor(Uk, dtype=torch.float32)
proj = torch.tensor(mu_u, dtype=torch.float32) + ((uk_t - torch.tensor(mu_u, dtype=torch.float32)) @ torch.tensor(Bpc.T, dtype=torch.float32)) @ torch.tensor(Bpc, dtype=torch.float32)
Td = TOK.float().clone(); Td[:, 6] = proj
Pc = prelude2(Td, Wp)
dEd = (Pc['E'] - E_ref_d).abs()
print(f'clone (PCA proj of u) at drug bundles: E change max {float(dEd.max()):.4f} mean {float(dEd.mean()):.5f}; '
      f'attn to z per head (mean over slots, bundles):', [round(float(Pc["att"][:, h, :4, 6].mean()), 3) for h in range(4)],
      ' attn to u:', [round(float(Pc["att"][:, h, :4, 5].mean()), 3) for h in range(4)])
```