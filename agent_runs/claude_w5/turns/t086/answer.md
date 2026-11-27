Run 81 shows that removing the gap term from ctx entirely gives the best fits for every decay strength, about 0.59–0.63 relative error. The documented decay (multiplier 1) is now among the best. So my gap term adds a wrong vector, and a second, larger error remains elsewhere.

Before going further I'll check where that remaining error sits. I'll group the bundles by their atom composition, for example glucose_finger alone, map_cuff alone, or several readings together, and compare the errors across groups. I'll also print the norm of my gap vector next to the norm of out, to see how much harm a wrong gap term can do. If one composition (say, bundles holding a particular code) is much worse, the embedding of that code is the culprit. If the error is similar everywhere, the problem is in the shared ctx pieces.

```python
from collections import defaultdict
def zp_by_comp(stays, build, max_n=400):
    errs = defaultdict(list)
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if aa or not wa: continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            zh = mlp_(build(out_p, h_p, dt, wa), 'z_post', 'gelu')[:32]; zt = Xt[r0_+k_, 960:]
            key_ = '+'.join(sorted(a['code'] for a in wa)) + f'|dt{int(round(dt))}'
            errs[key_].append(float(torch.norm(zh-zt)/torch.norm(zt)))
    return sorted([(k_, len(v_), round(float(np.median(v_)),3)) for k_, v_ in errs.items()], key=lambda t: -t[1])
bld_ng = lambda o,h,dt,wa: torch.cat([o + ln_(flow_c(h, dt, 1.0).mean(0),'core.out_norm'), E_(wa), U0])
for row_ in zp_by_comp(list(bundles_B.keys())[:30], bld_ng)[:16]: print(row_)
o_ex = Xt[5, :192]
print('norm out', float(torch.norm(o_ex)), 'norm gap(3600s)', float(torch.norm(G_(3600.0))), 'norm gap(300s)', float(torch.norm(G_(300.0))))
```