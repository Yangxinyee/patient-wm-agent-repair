Switching every MLP to GELU helps: error falls to 2.88 and my NLL to 1.486, against the tool's 1.4235. Some of my earlier SiLU choice was probably wrong.

There is still a gap of 0.06 nats. Next I check two things:
- the storage precision of the state file, since float16 rounding could cause part of the mismatch;
- an average over 32 prior z draws with GELU, which imitates the tool's 4-draw probability averaging. I also try the tanh-approximate GELU.

If sampling closes the NLL gap, the head re-implementation is good enough to calibrate the readout. If not, an input is still missing.

```python
print(S['x'].dtype, S['x'][0,:4])
def run10(f8, a='gelu', nz=0, ty=1, seed=0):
    torch.manual_seed(seed)
    hn = LN_(hfl.mean(1), 'core.out_norm'); b_ = OUT[P_] + hn
    ctx_ = b_ + Ma((torch.log(dt_)/10)[:,None], 'gap', a)
    zp = Ma(ctx_, 'z_prior', a); zmu, zls = zp[:, :32], zp[:, 32:]
    other = Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t]
    tr_ = Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    zs_list = [zmu] if nz==0 else [zmu + torch.exp(zls)*torch.randn_like(zmu) for _ in range(nz)]
    pacc = 0
    for z_ in zs_list:
        q_ = LN_(ctx_ + L(z_, 'z_head') + Wd['type_emb.weight'][ty], 'query_norm')
        pacc = pacc + torch.softmax(Ma(q_ + other, 'attr_heads.value_bin', a) + tr_, -1)
    p_ = (pacc/len(zs_list)).numpy()
    pm_ = (p_[:,1:]*cenmat).sum(1); nll_m = -np.log(p_[np.arange(len(p_)), ob_] + 1e-12)
    return pm_, p_, nll_m
for a_, nz in [('gelu',0),('gelu',4),('gelu',32),('gelut',0),('gelut',32)]:
    pm_, p_, nl_ = run10(fC, a_, nz)
    print(a_, nz, rep(pm_), round(float(nl_.mean()),4), 'corr', round(np.corrcoef(pm_, tgt)[0,1],4))
```