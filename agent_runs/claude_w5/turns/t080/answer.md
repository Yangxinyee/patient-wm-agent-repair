Run 75 narrows it down:
- The first 32 outputs of z_post are the posterior mean; the other ways of slicing the output are uncorrelated with the recorded z.
- The gap input log(dt in seconds)/10 is the best scaling.
- Skipping the time flow on h lowers the error, from 0.85 to 0.69, and raises the correlation from 0.55 to 0.79.

So my time flow decays h far too much. Either flow.a is not a log-rate per second, or the decay is defined differently (for example, exp(a) being the timescale rather than the rate).

I'll print the implied timescales and compare them with the documented range of 600 s to 2 years (vocab config tau_out_range_s). Then I'll test variants of the decay definition. A correct definition should push the z error clearly below the no-flow value.

```python
a_ = W['core.flow.a']
print('flow.a range', a_.min(), a_.max())
print('exp(a) range', np.exp(a_).min(), np.exp(a_).max(), ' exp(-a) range', np.exp(-a_).min(), np.exp(-a_).max())
print('quantiles exp(-a):', np.quantile(np.exp(-a_), [0, .1, .25, .5, .75, .9, 1]))
print('core.a_in range', W['core.a_in'].min(), W['core.a_in'].max(), 'delta_in range', W['core.delta_in'].min(), W['core.delta_in'].max())
def flow_variant(h_p, dt, mode):
    a_t = T_['core.flow.a']
    if mode == 'rate_s':   return h_p * torch.exp(-dt * torch.exp(a_t))
    if mode == 'tau_s':    return h_p * torch.exp(-dt / torch.exp(a_t))
    if mode == 'rate_h':   return h_p * torch.exp(-(dt/3600) * torch.exp(a_t))
    if mode == 'softplus': return h_p * torch.exp(-dt * Fnn.softplus(a_t))
    if mode == 'none':     return h_p
    if mode == 'tau_clip':
        tau = torch.clamp(torch.exp(-a_t), 600.0, 63115200.0); return h_p * torch.exp(-dt / tau)
def zpost_diag2(opt, stays, mode, act='gelu', max_n=150):
    errs, cors = [], []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if aa or not wa: continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            hf = flow_variant(h_p, dt, mode)
            ctx = out_p + ln_(hf.mean(0), 'core.out_norm') + mlp_(torch.tensor([math.log(dt)/10]), 'gap', act)
            e = setenc([atom_vec(a, opt) for a in wa], 'encode', act)
            zh = mlp_(torch.cat([ctx, e, torch.zeros(192)]), 'z_post', act)[:32]; zt = Xt[r0_+k_, 960:]
            errs.append(float(torch.norm(zh-zt)/torch.norm(zt))); cors.append(float(np.corrcoef(zh.numpy(), zt.numpy())[0,1]))
            if len(errs) >= max_n: break
        if len(errs) >= max_n: break
    return round(np.median(errs),4), round(np.median(cors),3)
for mode in ['rate_s','tau_s','rate_h','softplus','none','tau_clip']:
    print(mode, zpost_diag2(optA, stays_t, mode))
```