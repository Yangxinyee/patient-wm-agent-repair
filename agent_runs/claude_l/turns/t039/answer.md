The bundle parsing is exact for all 100 stays: bundle counts, atom counts and row order all match the tool. The z prediction is still off, by a median max error of 0.5 (GELU, log(dt)/10) to 0.9, even on measurement-only bundles where my context should be exact.

So z_post's output probably isn't the reported posterior mean by itself. A common design is for the posterior network to output a correction to the prior mean, or to be combined with the prior. I'll test whether the reported z equals:
- (a) prior mean + z_post output;
- (b) a precision-weighted combination of prior and z_post;
- (c) the z_post output alone (as now);
across activations and gap scalings. A near-exact match for one form would settle it.

```python
def zcands(s, k, act, gf, use_kq=True):
    rows = row_of[s]; prev = X[rows[k-1]]; t_prev = BUN[s][k-1][0]; t, atoms = BUN[s][k]
    dt = (t - t_prev)*3600
    out_p = torch.tensor(prev[:192]); h_p = torch.tensor(prev[192:960]).reshape(4,192)
    h_d = h_p * torch.exp(-dt*torch.exp(Wt['core.flow.a']))
    ctx = out_p + ln(h_d.mean(0), 'core.out_norm') + mlp(torch.tensor([gf(dt)], dtype=torch.float32), 'gap', act)
    meas = [a for a in atoms if not a['act']]; acts = [a for a in atoms if a['act']]
    e = set_encode(atoms_tensor(meas, use_kq), 'encode', act) if meas else torch.zeros(192)
    u = set_encode(atoms_tensor(acts, use_kq), 'encode_action', act) if acts else torch.zeros(192)
    po = mlp(torch.cat([ctx, e, u]), 'z_post', act); pr = mlp(ctx, 'z_prior', act)
    pm, pls = po[:32], po[32:]; qm, qls = pr[:32], pr[32:]
    prec_p, prec_q = torch.exp(-2*pls), torch.exp(-2*qls)
    c = {'post': pm, 'prior+post': qm+pm, 'precw': (pm*prec_p + qm*prec_q)/(prec_p+prec_q)}
    return c, torch.tensor(X[rows[k], 960:992]), len(meas), len(acts)
for actn in ['silu','gelu']:
    for gname, gf in [('log/10', lambda dt: np.log(dt)/10), ('logh/10', lambda dt: np.log(dt/3600)/10), ('log1p/10', lambda dt: np.log1p(dt)/10)]:
        E_ = {}
        for s in stays_b1[:8]:
            for k in range(1, len(BUN[s])):
                c, zt, nm, na = zcands(s, k, ACTS[actn], gf)
                if na==0 and nm>0:
                    for kk, v in c.items(): E_.setdefault(kk, []).append((v-zt).abs().max().item())
        print(actn, gname, {kk: (round(np.median(v),4), round(np.max(v),4)) for kk, v in E_.items()})
```