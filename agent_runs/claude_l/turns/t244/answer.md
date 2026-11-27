Copying u into the z token only at drug bundles is almost free: dNLL +0.00007 per reading, mean |Δmean| 0.002. Copying at reading bundles costs ten times more (0.0007) and is not needed for Stage 1. So the design is:
- **Drug bundles:** z token ≈ u, plus the residual-dose injection.
- **Reading bundles:** an unattended constant token t0.

In weights this needs a gate inside z_post. I'll use gelu units with a large offset driven by a "drug present" readout from u, minus a shared gate unit, so the copy and residual terms vanish at reading bundles.

This run does two things:
1. Check that a constant t0 at reading bundles (the original z_in bias) is as neutral as the original z tokens. Confirmed if dNLL stays near 1e-4.
2. Compute the token-to-next-reading Jacobians at drug bundles on the copy_drug baseline. I'm looking for a stable per-drug direction: high cosine between bundles, finger and lab aligned, little MAP cross-talk for insulin and the reverse for NE.

```python
import numpy as np, torch, time
t0_tok = Wc['z_in.bias'].clone()[None]
zf_t0 = lambda ctx, e, u, info: u if info is not None else t0_tok.expand(u.shape[0], -1)
dm, dn = [], []
for s in list(BUN)[:5]:
    Ra = tf_stay_z(Wc, BUN[s], None); Rb = tf_stay_z(Wc, BUN[s], zf_t0)
    dm += [abs(a[2]-b[2]) for a, b in zip(Ra, Rb)]; dn += [b[3]-a[3] for a, b in zip(Ra, Rb)]
print('t0=z_in.bias at reading, copy at drug: max|dmean| %.3f mean %.4f dNLL %.5f' % (max(dm), np.mean(dm), np.mean(dn)))
t0c = time.time(); JT = []
for s in list(BUN)[:20]: JT += [(s,)+r for r in jac_tok_stay(Wc, BUN[s], D_eps)]
print('n drug bundles', len(JT), 'time %.0fs' % (time.time()-t0c))
Jall = np.stack([r[7] for r in JT]); hi = np.array([r[2] for r in JT]); hn = np.array([r[3] for r in JT])
def cos(a, b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
for nm_, m, ci in [('ins-only', (hi>0)&(hn==0), (1,3)), ('ne-only', (hn>0)&(hi==0), (2,0)), ('both', (hi>0)&(hn>0), (1,2))]:
    if m.sum() < 3: print(nm_, 'n', m.sum()); continue
    Jm_ = Jall[m].mean(0)
    cons = np.mean([cos(Jall[i][:,ci[0]], Jm_[:,ci[0]]) for i in np.where(m)[0]])
    print(nm_, 'n', m.sum(), '|Jmean| [cuff,fing,art,lab]', np.round(np.linalg.norm(Jm_, axis=0), 2).tolist(),
          'cons %.2f' % cons, 'cos(c%d,c%d) %.2f' % (ci[0], ci[1], cos(Jm_[:,ci[0]], Jm_[:,ci[1]])))
```