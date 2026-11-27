Two findings change the plan:
- The core update is not weak: softplus(delta_in) ≈ 1 and a_bar ≈ 0.36, so h is largely rewritten every step. The tiny effect of prelude injections must come from LayerNorm and coda dilution, not from a slow integrator.
- z_prior has 29 dead hidden units, and z_head is a plain linear map from z into the view, where the leverage is about 13 mg/dL per unit.

So the design becomes three linked parts:
1. **Core write.** z_post spare units compute gated deviations. An attractor in the z token makes head 3 attend to z at drug bundles. Value-only z-token directions write the deviations into the state.
2. **Head read.** Dead z_prior units read that trace from the next context into a hijacked prior dim with tiny log-sd.
3. **View shift.** That dim's z_head column pushes the view along the glucose or MAP axis.

The core write does not need to be large; it needs to be distinguishable from the context's natural variation.

Next I'll measure that. I'll take the change in the next context produced by each of the 48 head-3 value directions under the attractor, at 25 insulin bundles. I'll compare it with the natural covariance of the context (ctx2), giving the best achievable signal-to-noise per unit value and its consistency across bundles. An SNR well above about 5 within the linear injection range confirms the route.

```python
def view_with(k, add):
    c, z, h, o = step_local(k, ztok_add=add); dt1 = dts[k+1]; return ctx_from(o, h*torch.exp(-dt1*torch.exp(fa)), dt1)
selS = ki[::50][:25]; Jc = np.zeros((len(selS),192,48))
for a_,k in enumerate(selS):
    vA = view_with(k, Gg*Ax)
    for m in range(48): Jc[a_,:,m] = ((view_with(k, Gg*Ax + s_*Gg*Wm[:,m]) - vA)/s_).numpy()
C2 = ctx2.float().numpy(); Sig = np.cov(C2[1:].T); lam = 1e-3*np.trace(Sig)/192; Si_ = np.linalg.inv(Sig + lam*np.eye(192))
Jb = Jc.mean(0); Mm = Jb.T@Si_@Jb; ev, evec = np.linalg.eigh(Mm); vstar = evec[:,-1]
print('top SNR per unit value: %.1f (2nd %.1f); |Jb v*| %.3f'%(np.sqrt(ev[-1]), np.sqrt(ev[-2]), np.linalg.norm(Jb@vstar)))
wread = Si_@Jb@vstar; sig_k = np.array([wread@(Jc[a_]@vstar) for a_ in range(len(selS))]); noise = np.sqrt(wread@Sig@wread)
print('per-bundle signal / noise: mean %.1f, CV %.2f, min %.1f'%(sig_k.mean()/noise, sig_k.std()/sig_k.mean(), sig_k.min()/noise))
cosJ = [float(np.dot(Jc[a_]@vstar, Jb@vstar)/np.linalg.norm(Jc[a_]@vstar)/np.linalg.norm(Jb@vstar)) for a_ in range(len(selS))]
print('cos(J_k v*, Jb v*) med %.2f min %.2f'%(np.median(cosJ), np.min(cosJ)))
```