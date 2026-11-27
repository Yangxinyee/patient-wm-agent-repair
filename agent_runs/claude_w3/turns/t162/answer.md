Each bundle is a small table (minute, code, kind, ekind, value, rate, dose), and `P0` holds all 181 weight tensors. I'll rebuild e and u through the atom-level path, `atom_vec` for each row then `set_enc_` with 'encode' or 'encode_action'. Rows with drug codes (8 and 9) go to the action encoder, all others to the measurement encoder. Then I'll rerun the z check on the same stay. A clear drop below 0.59 would show `enc_bundle` was the faulty piece.

```python
def enc2(g):
    va, vm = [], []
    for r in g.itertuples():
        v = atom_vec(int(r.code), int(r.kind), int(r.ekind), r.value, r.rate, r.dose, P0, {})
        (va if int(r.code) in (8, 9) else vm).append(v)
    z0 = torch.zeros(192, dtype=torch.float64)
    e = torch.as_tensor(set_enc_(vm, 'encode', P0), dtype=torch.float64) if vm else z0
    u = torch.as_tensor(set_enc_(va, 'encode_action', P0), dtype=torch.float64) if va else z0
    return e.reshape(-1), u.reshape(-1)
def zp_enc2(sid, kmax=60):
    m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(sid); errs = []
    for k in range(1, min(kmax, len(B))):
        e, u = enc2(B[k])
        dt = (hr_[k]-hr_[k-1])*3600.; hp = Xs[k-1,192:960].reshape(4,192)*torch.exp(-dt*torch.exp(fa_))
        ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
        zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
        errs.append(((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item())
    return np.round(np.percentile(errs, [10, 50, 90]), 3)
print(zp_enc2(s0_))
```