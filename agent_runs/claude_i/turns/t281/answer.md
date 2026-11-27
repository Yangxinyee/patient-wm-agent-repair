`step_local` fails for this particular bundle on an internal lookup, but it isn't needed here. The meaningful check is whether `step_u` reproduces the tool's recorded state for the bundle. So I'll drop the `step_local` comparison and compare `step_u` with the fwd_3 state directly, then run the doubling-SNR measurement as planned.

```python
k = int(ki[3]); h1, o1 = step_u(k, Uu_[k]); print('step_u vs tool: |h-HHt| %.2e  |o-OUTt| %.2e'%((h1-HH[k]).abs().max(), (o1-OUTt_[k]).abs().max()))
kk = [int(k) for k in ki if k+1<NBn and sid_[k+1]==sid_[k]]; CT = {}
for f in (0.5, 1.0, 2.0): CT[f] = torch.stack([ctx_next(*step_u(k, torch.as_tensor(u_for(k, fI=f)).float().reshape(192)), k) for k in kk]).numpy()
rk = np.array([INFO[k]['insulin'] for k in kk]); D2 = CT[2.0]-CT[1.0]; Dh = CT[0.5]-CT[1.0]
Sig2 = np.cov(CT[1.0].T); Si2 = np.linalg.inv(Sig2 + 1e-3*np.trace(Sig2)/192*np.eye(192))
wF = Si2@(D2/rk[:,None]).mean(0); nz_ = np.sqrt(wF@Sig2@wF); resp2 = D2@wF/nz_; resph = Dh@wF/nz_
print('n', len(kk), 'doubling: SNR per u/h mean %.2f (CV %.2f); halving per u/h %.2f; corr(resp2, rate) %.2f'%((resp2/rk).mean(), (resp2/rk).std()/(resp2/rk).mean(), (resph/(-0.5*rk)).mean(), np.corrcoef(resp2, rk)[0,1]))
print('|D2| med %.3f'%np.median(np.linalg.norm(D2,axis=1)))
```