`step_local` failed on a lookup because the bundle index was a numpy integer rather than a Python int. Earlier calls passed plain ints. I'll cast indices to int and rerun the same check and doubling-SNR measurement. The question is unchanged: does the model's own state carry a dose change clearly enough for a head-side readout?

```python
k = int(ki[3]); h1, o1 = step_u(k, Uu_[k]); c_, z_, h2, o2 = step_local(k)
print('check vs step_local: |dh| %.2e |do| %.2e ; vs tool |h-HHt| %.2e'%((h1-h2).abs().max(), (o1-o2).abs().max(), (h1-HH[k]).abs().max()))
kk = [int(k) for k in ki if k+1<NBn and sid_[k+1]==sid_[k]]; CT = {}
for f in (0.5, 1.0, 2.0): CT[f] = torch.stack([ctx_next(*step_u(k, torch.as_tensor(u_for(k, fI=f)).float().reshape(192)), k) for k in kk]).numpy()
rk = np.array([INFO[k]['insulin'] for k in kk]); D2 = CT[2.0]-CT[1.0]; Dh = CT[0.5]-CT[1.0]
Sig2 = np.cov(CT[1.0].T); Si2 = np.linalg.inv(Sig2 + 1e-3*np.trace(Sig2)/192*np.eye(192))
wF = Si2@(D2/rk[:,None]).mean(0); nz_ = np.sqrt(wF@Sig2@wF); resp2 = D2@wF/nz_; resph = Dh@wF/nz_
print('n', len(kk), 'doubling: SNR per u/h mean %.2f (CV %.2f); halving per u/h %.2f; corr(resp2, rate) %.2f'%((resp2/rk).mean(), (resp2/rk).std()/(resp2/rk).mean(), (resph/(-0.5*rk)).mean(), np.corrcoef(resp2, rk)[0,1]))
print('|D2| med %.3f  |u2-u1| med %.3f'%(np.median(np.linalg.norm(D2,axis=1)), np.median([float((torch.as_tensor(u_for(k,fI=2.0)).float().reshape(192)-Uu_[k]).norm()) for k in kk[:50]])))
```