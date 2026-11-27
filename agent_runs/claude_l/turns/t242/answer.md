That run failed on dtype only: the replica works in float64 (T) and I built the weight dict in float32. The measurement codes are 5, 6, 7 and 10. I'll map them to finger, lab, arterial and cuff from VOCAB so the Jacobian columns are labelled correctly.

The plan is unchanged: build the weights in dtype T, verify that copying u into the z token leaves predictions unchanged, then compute the token-to-next-reading Jacobians at drug bundles. Confirmation looks the same as before: a consistent per-drug direction moving the matching readings, with little cross-talk.

```python
import torch, math, numpy as np, time
Wc = {n: torch.as_tensor(load(n)).to(T) for n in NAMES}
cv = VOCAB.get('codes', None)
try:
    nm_of = {v: k for k, v in cv.items()} if isinstance(cv, dict) else {i: x for i, x in enumerate(cv)}
    print('codes4', [(c, nm_of.get(c)) for c in codes4])
except Exception as ex: print('codes vocab type', type(cv).__name__, str(cv)[:200])
s0 = list(BUN)[0]
Ra = tf_stay_z(Wc, BUN[s0], None); Rb = tf_stay_z(Wc, BUN[s0], copyz)
print('copy-u check: max |dmean| %.2e, dNLL %.2e' % (max(abs(a[2]-b[2]) for a, b in zip(Ra, Rb)), np.mean([b[3]-a[3] for a, b in zip(Ra, Rb)])))
usd = float(np.std(np.asarray(DA_u)[np.asarray(DA_meta)[:,4]>0], axis=1).mean()); eps_ = 0.05*usd
D_eps = torch.eye(192, dtype=T)*eps_
t0 = time.time(); JT = []
for s in list(BUN)[:20]: JT += [(s,)+r for r in jac_tok_stay(Wc, BUN[s], D_eps)]
print('n drug bundles', len(JT), 'time %.0fs' % (time.time()-t0), 'eps', round(eps_,4))
Jall = np.stack([r[7] for r in JT])
hi = np.array([r[2] for r in JT]); hn = np.array([r[3] for r in JT])
for nm_, m, ci in [('ins-only', (hi>0)&(hn==0), (0,1)), ('ne-only', (hn>0)&(hi==0), (2,3)), ('both', (hi>0)&(hn>0), (0,2))]:
    if m.sum() < 3: print(nm_, 'n', m.sum()); continue
    Jm_ = Jall[m].mean(0)
    cons = np.mean([cos(Jall[i][:,ci[0]], Jm_[:,ci[0]]) for i in np.where(m)[0]])
    print(nm_, 'n', m.sum(), '|Jmean| per code', np.round(np.linalg.norm(Jm_, axis=0), 2).tolist(), 'cos(bundle,mean) %.2f' % cons,
          'cos(c%d,c%d) %.2f' % (ci[0], ci[1], cos(Jm_[:,ci[0]], Jm_[:,ci[1]])))
```