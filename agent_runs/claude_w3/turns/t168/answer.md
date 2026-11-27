Using e = encode(empty set) for action-only bundles cuts their error from 0.35 to 0.12. So empty sets are encoded as the encoder of an empty set, not as zero. By symmetry, measurement-only bundles probably use u = encode_action(empty set) rather than zero. I'll test that on measurement bundles, with and without the time flow. A similar drop would confirm the convention.

```python
u_empty = LNd(MLP(LIN(torch.tensor([np.log1p(0.)]), 'encode_action.size'), 'encode_action.mlp2'), 'encode_action.norm')
def enc3(g):
    e, u = enc2(g)
    cs = set(g.code.astype(int))
    if cs <= {8, 9}: e = e_empty
    if not (cs & {8, 9}): u = u_empty
    return e, u
def zp_all3(flow=True, kmax=400, which='all'):
    errs = []
    for sid in np.unique(F4['stay_id']):
        m = F4['stay_id']==sid; Xs = X_[m]; hr_ = F4['hour'][m]; B = stay_bundles(int(sid))
        for k in range(1, min(kmax, len(B))):
            isact = bool(set(B[k].code.astype(int)) & {8, 9})
            if which == 'meas' and isact: continue
            e, u = enc3(B[k])
            dt = (hr_[k]-hr_[k-1])*3600.; hp = Xs[k-1,192:960].reshape(4,192)
            if flow: hp = hp*torch.exp(-dt*torch.exp(fa_))
            ctx = Xs[k-1,:192] + LNd(hp.mean(0), 'core.out_norm') + MLP(torch.tensor([np.log(dt)/10.]), 'gap')
            zm = MLP(torch.cat([ctx, e, u]), 'z_post')[:32]
            errs.append(((zm - Xs[k,960:]).norm()/Xs[k,960:].norm()).item())
    return len(errs), np.round(np.percentile(errs, [50, 90]), 3)
print(zp_all3(True, which='meas'), zp_all3(False, which='meas'), zp_all3(True))
```