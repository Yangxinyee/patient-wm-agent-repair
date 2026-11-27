Run 85 rules out the remaining simple hypotheses. The stored z is not the prior mean, it is not built from the current state, and dropping out or the normalised slot mean makes things worse. The best remains ctx = out + out_norm(mean of decayed h), with no gap term, at about 0.61. So ctx probably contains the documented pieces, but in a different mixture than I assume.

I'll now fit that mixture instead of guessing. I'll cache, for each measurement-only bundle, the three ctx pieces (previous out, normalised slot mean with and without decay, gap vector) and my encoding e. Then I'll grid-search a weight for each piece, plus the decay on or off. If some weighting gives a much smaller error, for example with the gap entering with a weight other than 1, that reveals how ctx is really formed. If the error stays near 0.6 at every weighting, the fault is in e after all.

```python
cache = []
for s_ in list(bundles_B.keys())[:20]:
    bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
    for k_ in range(1, len(bl_)):
        atoms = bl_[k_][1]
        wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
        if aa or not wa: continue
        prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
        dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
        cache.append((out_p, ln_(flow_c(h_p, dt, 1.0).mean(0),'core.out_norm'), ln_(h_p.mean(0),'core.out_norm'), G_(dt), E_(wa), Xt[r0_+k_, 960:]))
        if len(cache) >= 150: break
    if len(cache) >= 150: break
O_ = torch.stack([c[0] for c in cache]); HF_ = torch.stack([c[1] for c in cache]); HN_ = torch.stack([c[2] for c in cache])
GP_ = torch.stack([c[3] for c in cache]); EE_ = torch.stack([c[4] for c in cache]); ZT_ = torch.stack([c[5] for c in cache])
def batch_err(ctx):
    zh = mlp_(torch.cat([ctx, EE_, torch.zeros_like(EE_)], 1), 'z_post', 'gelu')[:, :32]
    return float(torch.median(torch.norm(zh-ZT_, dim=1)/torch.norm(ZT_, dim=1)))
grid = []
vals = [0, 0.5, 1, 1.5, 2]
for al in vals:
    for be in vals:
        for ga in vals:
            for fl in [0, 1]:
                H_ = HF_ if fl else HN_
                grid.append((round(batch_err(al*O_ + be*H_ + ga*GP_),4), al, be, ga, fl))
grid.sort(); print(grid[:10]); print('n', len(cache))
```