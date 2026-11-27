The head-3 injection route fails on signal-to-noise. The best value direction gives an SNR of only 0.5 per unit value change, with a per-bundle mean of 0.7 and a CV of 0.70. Since LayerNorm keeps the value change within a few units, z_prior would read mostly noise. The core dilutes anything written through the prelude.

A different question decides whether a head-only readout is viable: how distinguishable is the model's own state response to a real dose change? I'll write a step function that swaps in an edited insulin encoding (`u_for(k, fI)`) while keeping the correct memory of past readings, and first check it matches `step_local`. Then at all insulin bundles in the fwd_3 stays I'll compute the next context under insulin ×0.5, ×1 and ×2. The Fisher readout direction gives the SNR per unit rate change.
- An SNR per u/h of about 2 or more means dead z_prior units could read the dose change straight from the state at an acceptable factual-noise cost.
- Around 0.5 means the state simply doesn't carry the dose.

```python
sidT = torch.tensor(sid_); first = {}
for k in range(NBn):
    if sid_[k] not in first: first[sid_[k]] = k
OUTt_ = torch.as_tensor(OUTt).float(); Vd = {'incl':1,'unit':'s','age':'both'}
def step_u(k, u_new):
    hf = HH[k-1]*torch.exp(-dts[k]*torch.exp(fa)); ctx = ctx_from(OUTt_[k-1], hf, dts[k])
    z = zpost_mean(ctx, Ee_[k], u_new); idx_m = np.arange(max(first[sid_[k]], k+1-128), k+1)
    h, o = core_step(hf, Ee_[k], u_new, z, Ee_[idx_m], (tmin[k]-tmin[idx_m])*60.0, Vd); return h, o
k = ki[3]; h1, o1 = step_u(k, Uu_[k]); c_, z_, h2, o2 = step_local(k)
print('check vs step_local: |dh| %.2e |do| %.2e ; vs tool |h-HHt| %.2e'%((h1-h2).abs().max(), (o1-o2).abs().max(), (h1-HH[k]).abs().max()))
def ctx_next(h, o, k): dt1 = dts[k+1]; return ctx_from(o, h*torch.exp(-dt1*torch.exp(fa)), dt1)
kk = np.array([k for k in ki if k+1<NBn and sid_[k+1]==sid_[k]]); CT = {}
for f in (0.5, 1.0, 2.0): CT[f] = torch.stack([ctx_next(*step_u(k, torch.as_tensor(u_for(k, fI=f)).float().reshape(192)), k) for k in kk]).numpy()
rk = np.array([INFO[k]['insulin'] for k in kk]); D2 = CT[2.0]-CT[1.0]; Dh = CT[0.5]-CT[1.0]
Sig2 = np.cov(CT[1.0].T); Si2 = np.linalg.inv(Sig2 + 1e-3*np.trace(Sig2)/192*np.eye(192))
wF = Si2@(D2/rk[:,None]).mean(0); nz_ = np.sqrt(wF@Sig2@wF); resp2 = D2@wF/nz_; resph = Dh@wF/nz_
print('doubling: SNR per u/h  mean %.2f (CV %.2f);  halving per u/h %.2f;  corr(resp2, rate) %.2f'%((resp2/rk).mean(), (resp2/rk).std()/(resp2/rk).mean(), (resph/(-0.5*rk)).mean(), np.corrcoef(resp2, rk)[0,1]))
```