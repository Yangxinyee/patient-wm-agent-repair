The greedy search reached an error of 2.56, but with an implausible layout: two different age units, plus a "no previous reading" flag. It is probably fitting noise.

A likely cause of the remaining mismatch is that the tool averages bin probabilities over 4 z draws from the prior, while I use the prior mean.

Next I average my prediction over 32 prior draws of z, which should come close to the tool's expected output. I do this for three simple layouts that keep has_last in columns 0 and 3: age in hours, in minutes and in seconds. I also try a half-sized log-sd in case the 64 outputs are not [mean, log-sd].

- If the error falls well below 2.5 with a simple layout, sampling explained the gap and the simple layout is right.
- If not, the register layout is still wrong.

```python
torch.manual_seed(0)
def run4(f8, ty=1, nz=32, sdmode='exp'):
    ctx_ = base0 + gapv('log10')
    zp = M(ctx_, 'z_prior'); zmu, zls = zp[:, :32], zp[:, 32:]
    sd_ = torch.exp(zls) if sdmode=='exp' else torch.exp(0.5*zls)
    other = Wd['embed.code.weight'][codes_t] + L(torch.tensor(f8), 'last_numeric') + Wd['embed.bins.value.weight'][lb_t] + Wd['embed.bins.rate.weight'][0]
    tr_ = Wd['embed_transition.value_bin.weight'][codes_t*17 + lb_t]
    pacc = 0
    for _ in range(nz):
        z_ = zmu + sd_*torch.randn_like(zmu)
        q_ = LN_(ctx_ + L(z_, 'z_head') + Wd['type_emb.weight'][ty], 'query_norm')
        pacc = pacc + torch.softmax(M(q_ + other, 'attr_heads.value_bin') + tr_, -1)
    p_ = (pacc/nz).numpy()
    return (p_[:,1:]*cenmat).sum(1), p_
for nm3, asg in [('C_h', ['has','la','zl','has','zero','zero','zero','zero']),
                 ('C_m', ['has','la_m','zl','has','zero','zero','zero','zero']),
                 ('C_s', ['has','la_s','zl','has','zero','zero','zero','zero'])]:
    feats['la_m'] = np.log1p(fa4.age_s.values/60.0)/10*fa4.has.values
    for sdm in ['exp','half']:
        pm_, _ = run4(build(asg), sdmode=sdm)
        print(nm3, sdm, sc(pm_), fa4.assign(ad=np.abs(pm_-tgt)).groupby('event').ad.mean().round(2).to_dict())
```