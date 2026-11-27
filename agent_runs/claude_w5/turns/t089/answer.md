Run 84 confirms the documented numeric column order: [z_value, has_value] in columns 0 and 1 fits best for both codes. The atom embedding therefore looks as documented, yet errors of about 0.5–0.9 remain. Either the stored z is not exactly the quantity I assume, or the context it uses is built from a different state than I assume.

I'll test four hypotheses directly on measurement-only bundles:
1. The stored z is the prior mean rather than the posterior mean.
2. ctx is built from the state after this bundle instead of the previous one.
3. The time decay is applied to out as well.
4. The z context uses the previous step's ctx pieces in another combination.

A hypothesis that brings the error near zero would identify the stored quantity. If none does, the remaining mismatch must be in the set encoder's internals.

```python
def zp_hyp(stays, mode, max_n=120):
    errs = []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if aa or not wa: continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            cur = Xt[r0_+k_]; out_c = cur[:192]; h_c = cur[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            hf = flow_c(h_p, dt, 1.0); e = E_(wa)
            if mode == 'prior':       z_ = mlp_(out_p + ln_(hf.mean(0),'core.out_norm') + G_(dt), 'z_prior', 'gelu')[:32]
            elif mode == 'prior_ng':  z_ = mlp_(out_p + ln_(hf.mean(0),'core.out_norm'), 'z_prior', 'gelu')[:32]
            elif mode == 'cur_state': z_ = mlp_(torch.cat([out_c + ln_(h_c.mean(0),'core.out_norm') + G_(dt), e, U0]), 'z_post', 'gelu')[:32]
            elif mode == 'cur_ng':    z_ = mlp_(torch.cat([out_c + ln_(h_c.mean(0),'core.out_norm'), e, U0]), 'z_post', 'gelu')[:32]
            elif mode == 'out_only_flowed': z_ = mlp_(torch.cat([ln_(hf.mean(0),'core.out_norm') + G_(dt), e, U0]), 'z_post', 'gelu')[:32]
            elif mode == 'hn_only_ng': z_ = mlp_(torch.cat([ln_(hf.mean(0),'core.out_norm'), e, U0]), 'z_post', 'gelu')[:32]
            elif mode == 'base_ng':   z_ = mlp_(torch.cat([out_p + ln_(hf.mean(0),'core.out_norm'), e, U0]), 'z_post', 'gelu')[:32]
            zt = Xt[r0_+k_, 960:]
            errs.append(float(torch.norm(z_-zt)/torch.norm(zt)))
            if len(errs) >= max_n: break
        if len(errs) >= max_n: break
    return round(float(np.median(errs)),3)
for mode in ['prior','prior_ng','cur_state','cur_ng','out_only_flowed','hn_only_ng','base_ng']:
    print(mode, zp_hyp(stays_t, mode))
```