Run 76 shows that the flow timescales are as documented: exp(−a) runs from about 590 s to 6.6e7 s. Yet leaving the time flow out still fits better: error 0.69 with correlation 0.79, against 0.85 with the documented decay. Neither is close to exact, so a larger structural mistake is probably hiding behind the flow question. The most likely one is the order of the three blocks fed to z_post. The document says [ctx, e_k, u_k], but if the real order differs, every downstream comparison would look partly right and partly wrong, as it does now.

So I'll test all six orders of [ctx, e, u], each with and without the time flow. If one order brings the error near zero, the structure is settled, and the flow choice can then be decided cleanly. If none does, the error is inside ctx or e itself, and I'll check the sensitivity to each piece next.

```python
def zpost_diag3(opt, stays, mode, order, act='gelu', max_n=120):
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
            parts = {'c': ctx, 'e': e, 'u': torch.zeros(192)}
            zh = mlp_(torch.cat([parts[o] for o in order]), 'z_post', act)[:32]; zt = Xt[r0_+k_, 960:]
            errs.append(float(torch.norm(zh-zt)/torch.norm(zt))); cors.append(float(np.corrcoef(zh.numpy(), zt.numpy())[0,1]))
            if len(errs) >= max_n: break
        if len(errs) >= max_n: break
    return round(np.median(errs),4), round(np.median(cors),3)
for order in itertools.permutations('ceu'):
    print(''.join(order), 'flow', zpost_diag3(optA, stays_t, 'rate_s', order), 'noflow', zpost_diag3(optA, stays_t, 'none', order))
Wz = W['z_post.0.weight']
print('col-block norms of z_post.0:', [round(float(np.linalg.norm(Wz[:, i*192:(i+1)*192])),2) for i in range(3)])
```