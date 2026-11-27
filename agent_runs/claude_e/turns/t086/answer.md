The head's input q encodes the dose deviations poorly: cross-validated R² is only 0.2 for the combined target, 0.24 for lag 1, 0.16 for lag 2 and about 0 beyond lag 3. A linear head-level readout from q therefore cannot implement the correction as it stands.

The head only sees the slot-mean of h (through `out` and the time-decayed term) plus z noise. The per-slot information is lost. Earlier I found within-tier dose decodable from the full state with R² 0.68.

Next I will measure where the dose-deviation lags live in the state at the previous bundle: `out`, the slot-mean of h, the individual slots, and the z posterior. If the information sits in individual slots but not in the mean, I will have to route it into the slot-mean or `out`, which means editing the core. If the slot-mean has it but q does not, the loss comes from z noise or normalisation, and a different readout might work.

```python
def state_parts(Xs, R_):
    xs = Xs[torch.as_tensor(R_.prev.values)].numpy()
    out = xs[:, :192]; h = xs[:, 192:960]; hm = h.reshape(-1,4,192).mean(1); zp = xs[:, 960:]
    return {'out': out, 'hmean': hm, 'h_all': h, 'zpost': zp, 'all': xs}
PA_ = state_parts(XA, RA[gA]); P1_ = state_parts(X1, R1[g1m])
for nm in ['out','hmean','h_all','zpost','all']:
    F = np.vstack([PA_[nm], P1_[nm]])
    F = (F - F.mean(0))/ (F.std(0)+1e-6)
    r = [round(cv_r2(F, Lg[:,j], grp, 10.0)[0],2) for j in range(6)]
    rd = round(cv_r2(F, Dt, grp, 10.0)[0],3)
    print(f'{nm:6s} dims {F.shape[1]:4d}  per-lag dev R2 {r}  D_t R2 {rd}')
```