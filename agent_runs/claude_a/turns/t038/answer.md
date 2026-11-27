Setting the `type_emb` index to 1 for measurement atoms cut the error a lot: glucose RMSE fell from 11 to 7.0 and MAP from 5.4 to 2.0, with GELU and layout L2. The 15 type rows are probably 3 atom kinds × 5 event kinds, so a WORLD measurement is type 1. A WORLD drug atom would then be type 2.

The age unit is still ambiguous: glucose prefers hours and MAP prefers seconds. Also, the z prior has an sd of about 0.9, so the model's own 4-draw average is noisy around the expectation. My prior-mean shortcut could explain part of the remaining gap.

Next I will compare several z treatments against the reported means: the prior mean, a 64-draw expectation, and 4-draw averages. At the same time I will try age in seconds, minutes and hours, and the two bin-edge conventions. If the 4-draw scatter alone accounts for the remaining RMSE, the heads are effectively correct. If a clear gap remains, a term is still missing.

```python
def head_probs(act, dt_s, layout, age_mult, side, nz, seed=0, kq=None, zmode='sample'):
    A = acts[act]
    hdec = HS[kprev] * torch.exp(-dt_s[:, None, None] * torch.exp(P['core.flow.a']))
    g = mlp((torch.log(dt_s)/10)[:, None], 'gap', A)
    ctx = OUT[kprev] + ln(hdec.mean(1), 'core.out_norm') + g
    pr = mlp(ctx, 'z_prior', A); mu, ls = pr[:, :32], pr[:, 32:]
    ag = torch.tensor(np.log1p(age_h*age_mult)/10, dtype=torch.float32)
    hz = torch.tensor(has); zz = torch.tensor(zv); zero = torch.zeros_like(hz)
    f = torch.stack([hz, ag, zz, hz, zero, zero, zero, zero], 1)
    vb_ = np.zeros(len(R), np.int64)
    for ev in R.event.unique():
        m = (R.event == ev).values & (has > 0)
        e = num[(ev, 'value')]; vb_[m] = 1 + np.searchsorted(np.array(e['fine_bin_edges']), R.lastv.values[m], side=side)
    vbt = torch.tensor(vb_)
    extra = lin(f, 'last_numeric') + P['embed.bins.value.weight'][vbt] + P['embed.code.weight'][codes_]
    if kq is not None: extra = extra + P['embed.kq.weight'][kq]
    gen = torch.Generator().manual_seed(seed)
    ps = 0
    for i in range(nz):
        z = mu if zmode == 'mean' else mu + torch.exp(ls) * torch.randn(mu.shape, generator=gen)
        view = ctx + lin(z, 'z_head')
        q = ln(view + P['type_emb.weight'][1], 'query_norm')
        lg = mlp(q + extra, 'attr_heads.value_bin', A) + P['embed_transition.value_bin.weight'][codes_*17 + vbt]
        ps = ps + torch.softmax(lg, -1)
    return ps / nz, vbt
def score(p):
    m_ = (p[:, 1:] * centM[codes_]).sum(1); e_ = m_ - obs_pm
    return round(e_[torch.tensor(gm)].pow(2).mean().sqrt().item(), 3), round(e_[torch.tensor(mm)].pow(2).mean().sqrt().item(), 3), m_
# 1) z treatment
p_mean, _ = head_probs('gelu', dts_opts['s'], 'L2', 1.0, 'left', 1, zmode='mean')
p_exp, _ = head_probs('gelu', dts_opts['s'], 'L2', 1.0, 'left', 64)
p_4a, _ = head_probs('gelu', dts_opts['s'], 'L2', 1.0, 'left', 4, seed=1)
p_4b, _ = head_probs('gelu', dts_opts['s'], 'L2', 1.0, 'left', 4, seed=2)
s_mean, s_exp, s_4a, s_4b = score(p_mean), score(p_exp), score(p_4a), score(p_4b)
print('vs obs: mean', s_mean[:2], 'exp64', s_exp[:2], '4a', s_4a[:2], '4b', s_4b[:2])
d_ = s_4a[2] - s_exp[2]; print('4-sample noise rmse glu', d_[torch.tensor(gm)].pow(2).mean().sqrt().item(), 'map', d_[torch.tensor(mm)].pow(2).mean().sqrt().item())
# 2) age units / bin side / kq with expectation over z
for age_mult, side, kq in itertools.product([1.0, 60.0, 3600.0], ['left', 'right'], [None, 0, 1]):
    p_, _ = head_probs('gelu', dts_opts['s'], 'L2', age_mult, side, 32)
    print(age_mult, side, kq, score(p_)[:2]) if kq is None else None
```