The prelude copy is exact (maximum error 1e-6). But the slot FFN is not a good place to compute the surprise:
- The slot residual stream has a per-dimension spread of about 62. It is dominated by the state, which swamps the u signal.
- So the insulin rate is only partly readable from the FFN input (R² 0.72, mixed with what the state already predicts).
- The insulin-present flag is not linearly separable from measurement bundles: the lowest positive score (0.25) is below the highest negative (0.66).
- All four slots behave the same.

I am dropping the FFN route.

A key realisation: the posterior z is unused at evaluation. The heads draw z from the prior, and the prelude's slots give the z token essentially zero attention. So z_post and z_in are free to repurpose, and z_post is the one network that sees [ctx, e_k, u_k] cleanly.

New mechanism, "z-clone":
- At drug bundles, make the z token an exact copy of the u token plus a small surprise term. The surprise term goes along a direction that does not change keys.
- Heads 1 to 3, which attend to u with weight about 1, would then split exactly 50/50 between two tokens with identical keys. Their outputs are unchanged, except for half the surprise term's value.
- At non-drug bundles the z token stays as it was, receiving no attention.

This only works if z_in(z), with z of 32 dimensions, can reproduce u_k. That requires the drug-bundle u vectors to lie, to good precision, in an affine subspace of about 29 dimensions. That is what I test now, with a PCA of the u-vectors over the insulin×NE rate grid I built earlier and the actual drug-bundle u's.

What the result would mean:
- Residual below about 0.1% of the variance at about 25 components: the mechanism is viable.
- Large residuals: it is not, and I fall back to rerouting head 1's value path.

```python
import numpy as np
aG = np.asarray(aG2, float); bG = np.asarray(bG2, float); UG = np.asarray(UG2, float)
print('grid pairs', UG.shape, 'ins rate range', aG.min(), aG.max(), 'NE rate range', bG.min(), bG.max(),
      'n both-zero', int(((aG == 0) & (bG == 0)).sum()), 'n ins-only', int(((aG > 0) & (bG == 0)).sum()),
      'n ne-only', int(((aG == 0) & (bG > 0)).sum()), 'n both', int(((aG > 0) & (bG > 0)).sum()))
Uk = uk.numpy().astype(float) if hasattr(uk, 'numpy') else np.asarray(uk, float)
uc = u_const.numpy().astype(float)
drug = (aG > 0) | (bG > 0)
Ud = UG[drug]
mu = Ud.mean(0)
Uc_ = Ud - mu
_, svals, Vt = np.linalg.svd(Uc_, full_matrices=False)
tot = (svals**2).sum()
print('u norm typical', np.linalg.norm(Ud, axis=1).mean().round(2), ' |u_const-mean|', np.linalg.norm(uc - mu).round(2),
      ' total var per row', (tot / len(Ud)).round(3))
for r in [5, 10, 15, 20, 25, 28, 29, 31]:
    B = Vt[:r]
    res_g = Uc_ - Uc_ @ B.T @ B
    res_k = (Uk - mu) - (Uk - mu) @ B.T @ B
    print(f'r={r:2d}: grid resid var frac {(res_g**2).sum()/tot:.2e}, max row resid norm {np.linalg.norm(res_g,axis=1).max():.3f} | '
          f'data u resid rms norm {np.sqrt((res_k**2).sum(1).mean()):.4f} max {np.linalg.norm(res_k,axis=1).max():.3f}')
```