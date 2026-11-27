Run 77 confirms the documented input order [ctx, e, u]: every other order is far worse. But even the best order still has an error of about 0.69–0.82, so a substantial piece is wrong. It is either part of ctx (out, the normalised slot mean, the gap term, the time flow) or my bundle encoding e.

I'll do a sensitivity check that changes one piece at a time:
- drop or zero e;
- remove each ctx term separately;
- try the set encoder with the size term placed differently, or without its final LayerNorm.

Whichever change barely moves the error marks a piece that is either irrelevant or already right. A change that makes the error drop sharply exposes the faulty piece.

```python
def zp_generic(stays, build, max_n=120):
    errs, cors = [], []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if aa or not wa: continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            zin = build(out_p, h_p, dt, wa)
            zh = mlp_(zin, 'z_post', 'gelu')[:32]; zt = Xt[r0_+k_, 960:]
            errs.append(float(torch.norm(zh-zt)/torch.norm(zt))); cors.append(float(np.corrcoef(zh.numpy(), zt.numpy())[0,1]))
            if len(errs) >= max_n: break
        if len(errs) >= max_n: break
    return round(np.median(errs),4), round(np.median(cors),3)
G_ = lambda dt: mlp_(torch.tensor([math.log(dt)/10]), 'gap', 'gelu')
def setenc_v(vecs, p, variant):
    n = len(vecs); m1 = mlp_(torch.stack(vecs), p+'.mlp1', 'gelu')
    sz = lin_(torch.tensor([math.log(1+n)]), p+'.size')
    if variant == 'std':     return ln_(mlp_(m1.sum(0)/math.sqrt(n) + sz, p+'.mlp2', 'gelu'), p+'.norm')
    if variant == 'mean':    return ln_(mlp_(m1.mean(0) + sz, p+'.mlp2', 'gelu'), p+'.norm')
    if variant == 'sum':     return ln_(mlp_(m1.sum(0) + sz, p+'.mlp2', 'gelu'), p+'.norm')
    if variant == 'post_sz': return ln_(mlp_(m1.sum(0)/math.sqrt(n), p+'.mlp2', 'gelu') + sz, p+'.norm')
    if variant == 'res':     x_ = m1.sum(0)/math.sqrt(n) + sz; return ln_(x_ + mlp_(x_, p+'.mlp2', 'gelu'), p+'.norm')
    if variant == 'nonorm':  return mlp_(m1.sum(0)/math.sqrt(n) + sz, p+'.mlp2', 'gelu')
    if variant == 'res1':    v_ = torch.stack(vecs); m1r = v_ + m1; x_ = m1r.sum(0)/math.sqrt(n) + sz; return ln_(x_ + mlp_(x_, p+'.mlp2', 'gelu'), p+'.norm')
E_ = lambda wa, variant='std': setenc_v([atom_vec(a, optA) for a in wa], 'encode', variant)
U0 = torch.zeros(192)
tests = {
 'full_noflow':  lambda o,h,dt,wa: torch.cat([o + ln_(h.mean(0),'core.out_norm') + G_(dt), E_(wa), U0]),
 'e_zero':       lambda o,h,dt,wa: torch.cat([o + ln_(h.mean(0),'core.out_norm') + G_(dt), torch.zeros(192), U0]),
 'ctx_out_only': lambda o,h,dt,wa: torch.cat([o, E_(wa), U0]),
 'ctx_no_gap':   lambda o,h,dt,wa: torch.cat([o + ln_(h.mean(0),'core.out_norm'), E_(wa), U0]),
 'ctx_no_hn':    lambda o,h,dt,wa: torch.cat([o + G_(dt), E_(wa), U0]),
}
for nm_, fn_ in tests.items(): print(nm_, zp_generic(stays_t, fn_))
for var in ['mean','sum','post_sz','res','nonorm','res1']:
    print('setenc', var, zp_generic(stays_t, lambda o,h,dt,wa, var=var: torch.cat([o + ln_(h.mean(0),'core.out_norm') + G_(dt), E_(wa, var), U0])))
```