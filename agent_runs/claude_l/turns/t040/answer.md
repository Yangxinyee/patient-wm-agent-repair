Neither prior-plus-posterior nor precision weighting reproduces the reported z (the best median error is still about 0.46), so the mismatch is elsewhere.

One unstated convention could matter: the activation might differ between submodules (for example gap, the set encoders and z_post), and the dt unit inside log(dt)/10 might be minutes rather than seconds or hours.

I'll run a full grid on measurement-only bundles:
- activation for gap × encoder × z_post, over {silu, gelu, relu, tanh};
- dt unit (s, min, h);
- with or without kq.

A combination with near-zero error would settle these conventions. If none comes close, the missing piece is structural, probably in the encoder or the atom vector.

```python
ACTS2 = {'silu': F.silu, 'gelu': F.gelu, 'relu': F.relu, 'tanh': torch.tanh}
samples = []
for s in stays_b1[:6]:
    for k in range(1, len(BUN[s])):
        t, atoms = BUN[s][k]; meas = [a for a in atoms if not a['act']]; acts = [a for a in atoms if a['act']]
        if acts or not meas: continue
        rows = row_of[s]; prev = X[rows[k-1]]; dt = (t - BUN[s][k-1][0])*3600
        out_p = torch.tensor(prev[:192]); h_p = torch.tensor(prev[192:960]).reshape(4,192)
        h_d = h_p * torch.exp(-dt*torch.exp(Wt['core.flow.a']))
        samples.append(dict(base=out_p + ln(h_d.mean(0), 'core.out_norm'), dt=dt, meas=meas, zt=torch.tensor(X[rows[k], 960:992])))
print(len(samples))
results = []
for kq in [True, False]:
    for ae in ['silu','gelu','relu']:
        E_list = [set_encode(atoms_tensor(sm['meas'], kq), 'encode', ACTS2[ae]) for sm in samples]
        for ag in ACTS2:
            for unit_name, div in [('s',1),('min',60),('h',3600)]:
                G_list = [mlp(torch.tensor([np.log(sm['dt']/div)/10], dtype=torch.float32), 'gap', ACTS2[ag]) for sm in samples]
                for az in ACTS2:
                    errs = [ (mlp(torch.cat([sm['base']+G, E, torch.zeros(192)]), 'z_post', ACTS2[az])[:32]-sm['zt']).abs().max().item() for sm,E,G in zip(samples,E_list,G_list)]
                    results.append((np.median(errs), kq, ae, ag, unit_name, az))
results.sort(key=lambda r: r[0]); print(results[:10])
```