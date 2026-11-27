The error is about the same size for every bundle type: 0.04–0.06 per dimension whatever the atom count or codes. So the set encoder and per-code embeddings are probably not the culprit. The context vector (previous out + normed decayed h + gap term) is more suspect.

To test the gap term directly, I'll treat its input as a free scalar for each bundle and scan it. If the best value sits consistently near ln(dt)/10, about 0.81 for a 55-minute gap, the gap is right and the fault is elsewhere. If it sits near a different function of dt, such as log10(dt)/10 ≈ 0.35, or the minimum error stays high wherever it sits, I'll know which piece to change.

```python
def err_at(test, g_in, eatoms_fn=meas_atom2):
    r_, rp, dt, ms_ = test
    e_ = encode([eatoms_fn(c, v) for c, v in ms_])
    out_p = Xs[rp, :192]; h_p = Xs[rp, 192:960].reshape(4,192)
    dec = torch.exp(-dt*torch.exp(Wt['core.flow.a']))
    ctx = out_p + ln((h_p*dec).mean(0), 'core.out_norm') + mlp(torch.tensor([g_in]), 'gap')
    zp = mlp(torch.cat([ctx, e_, torch.zeros(192)]), 'z_post')[:32]
    return (zp - Xs[r_, 960:992]).abs().mean().item()
grid = np.linspace(-1, 2, 61)
for t_ in tests[:12]:
    errs = [err_at(t_, g) for g in grid]
    i_ = int(np.argmin(errs))
    print('dt %.0f  ln(dt)/10=%.3f  best g=%.3f  min err %.4f  err@ln/10 %.4f' % (t_[2], math.log(t_[2])/10, grid[i_], errs[i_], err_at(t_, math.log(t_[2])/10)))
```