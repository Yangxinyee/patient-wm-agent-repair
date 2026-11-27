Run 102 confirms the rebuild almost everywhere:
- **Non-drug bundles:** the z posterior matches to 7e-6.
- **First bundle:** the initial state is zero h, with out equal to the normalisation of zero and a gap of about 1 second.
- **Drug bundles:** a few are still slightly off, with a maximum error of 0.076.

Drug rates and doses often fall exactly on fine-bin edges: insulin rates come in 0.5 steps, and norepinephrine at its cap of 1.0 has a dose of 4.8, which equals several edges. So the residual is probably a tie-breaking rule that differs for some attributes. I'll list the worst drug bundles with their rates and doses, then test bin-side rules per attribute and per drug. The rule that brings the maximum error to zero settles it.

```python
worst = []
for s_ in list(bundles_B.keys())[:30]:
    bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
    for k_ in range(1, len(bl_)):
        if not any(a['code'] in ACTION_CODES for a in bl_[k_][1]): continue
        e, u, _ = enc_bundle(bl_[k_][1])
        prev = Xt[r0_+k_-1]; dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
        ctx = prev[:192] + ln_(flow_c(prev[192:960].reshape(4,192), dt, 1.0).mean(0),'core.out_norm') + G_in(dt, 's10')
        zh = mlp_(torch.cat([ctx, e, u]), 'z_post', 'gelu')[:32]; zt = Xt[r0_+k_, 960:]
        er = float(torch.norm(zh-zt)/torch.norm(zt))
        worst.append((er, [(a['code'][:3], a['kind'], a.get('rate'), a.get('dose')) for a in bl_[k_][1] if a['kind'] == 0 and a['code'] in ACTION_CODES]))
worst.sort(key=lambda t: -t[0])
for w_ in worst[:12]: print(round(w_[0],4), w_[1])
print('n>1e-4:', sum(1 for w_ in worst if w_[0] > 1e-4), 'of', len(worst))
```