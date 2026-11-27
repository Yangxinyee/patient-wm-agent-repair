At the level of E the SNR is about seven times better than at the base. Decode noise is 0.20 U/h per unit token norm for insulin and 0.18 for NE, with the same values in every slot.

E itself barely varies across bundles: per-coordinate sd 0.19 on a vector of norm 14. That quietness is what makes it a good place to read the injected signal. The per-bundle gain varies by about ±38% for insulin and ±18% for NE. That adds noise but no bias.

The design is now:
1. **z_post / z_in:** at drug bundles the z token is a copy of u plus v·r, using the SNR-optimal token direction. At reading bundles it is an unattended constant t0.
2. **`core.b_in` rank-2 term:** κ·m·qᵀ maps E's quiet direction q (q orthogonal to the slot means) onto state directions m that move glucose (or MAP) at the next reading.
3. **Calibration:** set the κ gains with offline rollouts.

Before building it I need the costs of the parts that must be neutral, measured on 8 tuning stays:
- an approximate PCA copy of u through the 32-dim z bottleneck (K = 28 components, fitted on recorded and edited u's), with the original z elsewhere;
- two candidates for t0 at reading bundles: the mean of the original reading-bundle z tokens, and z_in.bias, which cost about 0.0002 earlier.

Copy error of a few percent and dNLL ≲ 1e-4 for both parts would confirm the base is neutral enough.

```python
import numpy as np, torch
Upool = np.vstack([np.asarray(DA_u)[::3], SYu]).astype(np.float64); mu_u = Upool.mean(0)
_, sv_, Vt_u = np.linalg.svd(Upool - mu_u, full_matrices=False)
for K in [16, 20, 24, 28]:
    P = Vt_u[:K]; R_ = (Upool-mu_u) - ((Upool-mu_u) @ P.T) @ P
    print('K', K, 'rel copy err mean %.4f max %.4f' % ((np.linalg.norm(R_, axis=1)/np.linalg.norm(Upool-mu_u, axis=1)).mean(), (np.linalg.norm(R_, axis=1)/np.linalg.norm(Upool-mu_u, axis=1)).max()))
P28 = torch.tensor(Vt_u[:28], dtype=T); mu_ut = torch.tensor(mu_u, dtype=T)
def copyK(u): return mu_ut + ((u - mu_ut) @ P28.T) @ P28
toks = []
def grab(ctx, e, u, info):
    zt = ztok_orig(Wc, ctx, e, u)
    if info is None: toks.append(zt[0].clone())
    return zt
for s in list(BUN)[20:25]: tf_stay_z(Wc, BUN[s], grab)
t0b = torch.stack(toks).mean(0)[None]; print('n reading toks', len(toks))
def zA(ctx, e, u, info): return copyK(u) if info is not None else ztok_orig(Wc, ctx, e, u)
def zB(ctx, e, u, info): return ztok_orig(Wc, ctx, e, u) if info is not None else t0b.expand(u.shape[0], -1)
def zC(ctx, e, u, info): return ztok_orig(Wc, ctx, e, u) if info is not None else t0_tok.expand(u.shape[0], -1)
for nm_, zf in [('PCA28 copy at drug', zA), ('t0=mean orig at reading', zB), ('t0=z_in.bias at reading', zC)]:
    dm, dn = [], []
    for s in list(BUN)[:8]:
        Ra = tf_stay_z(Wc, BUN[s], None); Rb = tf_stay_z(Wc, BUN[s], zf)
        dm += [abs(a[2]-b[2]) for a, b in zip(Ra, Rb)]; dn += [b[3]-a[3] for a, b in zip(Ra, Rb)]
    print(nm_, 'max|dmean| %.3f mean %.4f dNLL %.5f' % (max(dm), np.mean(dm), np.mean(dn)))
```