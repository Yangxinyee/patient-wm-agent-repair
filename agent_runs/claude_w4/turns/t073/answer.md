The encode_action replica is now exact for every drug bundle type, so I can compute u_k for any dose, including the doubled and halved ones in the evaluation:
- **NE atom:** unit row 4 and rate_unit row 3, with dose exactly 4.8 × rate (error about 1e-6).
- **Combined insulin + NE bundles:** match with error 0.0.

The next missing pieces are the other two inputs to z_post: ctx, built from the previous state, and e_k, from the measurement encoder. With those I can reproduce the z posterior on CPU, and so design repurposed z_post hidden units without disturbing the existing ones. I'll start with ctx, testing it against the probe's 4 exact ctx coordinates from fwd_4. The formula is

ctx = out_prev + out_norm(mean_slots(h_prev · exp(−dt·exp(flow.a)))) + gap(log dt / 10)

I'll try dt in seconds or in hours, log dt versus log(1 + dt), and print the decay timescales to confirm the units. One variant with error near 1e-4 confirms the ctx formula. If no variant matches, the context includes something else.

```python
ta = load('core.flow.a'); tau = 1/torch.exp(ta)
print('flow tau range (s):', float(tau.min()), float(tau.max()))
OUT4 = torch.tensor(X4[:, :192]); H4 = torch.tensor(X4[:, 192:960]).reshape(-1,4,192)
ctx_dims = [j for (j,b,c) in probe_cols if b=='ctx']
same_prev = np.r_[False, sid4[1:]==sid4[:-1]]
idx = np.where(same_prev)[0][:800]
dt_h = torch.tensor(hr4[idx] - hr4[idx-1], dtype=torch.float32)
def gapf(x): return load('gap.2.weight') @ gelu(load('gap.0.weight')[:,0][:,None]*x[None,:] + load('gap.0.bias')[:,None]) + load('gap.2.bias')[:,None]
onw, onb = load('core.out_norm.weight'), load('core.out_norm.bias')
for unit_dt, scale in [('s',3600.),('h',1.)]:
    for lgv in ['log','log1p']:
        dt = dt_h*scale
        hdec = H4[idx-1] * torch.exp(-dt[:,None,None]*torch.exp(ta)[None,None,:]) if unit_dt=='s' else H4[idx-1]*torch.exp(-dt[:,None,None]*3600*torch.exp(ta)[None,None,:])
        lg = (torch.log(dt) if lgv=='log' else torch.log1p(dt))/10
        ctxc = OUT4[idx-1] + LNf(hdec.mean(1), onw, onb) + gapf(lg).T
        err_ = (ctxc[:, ctx_c] - torch.tensor(Z4[idx][:, ctx_dims])).abs()
        print(unit_dt, lgv, 'max err', float(err_.max()), 'mean', float(err_.mean()))
```