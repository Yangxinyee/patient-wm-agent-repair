Run 79 rules out the gap activation: no choice gets the error below about 0.7. So the gap network is probably not the main fault, and my assumptions about the states file deserve suspicion. One possibility is that the z posterior mean stored in row k belongs to a different bundle, the next or the previous one. If so, every comparison so far was misaligned, which would explain moderate correlation without an exact match anywhere.

I'll test row offsets (−1, 0, +1) for the stored z. I'll also check whether the error comes from a simple per-dimension scale, by regressing each recorded z dimension on my value across bundles. A clean offset match would settle it. A high per-dimension correlation with a slope far from 1 would instead suggest a wrongly scaled input such as dt.

```python
def zp_collect(stays, build, max_n=300):
    ZH, ZT, ZTn, ZTp = [], [], [], []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)-1):
            atoms = bl_[k_][1]
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if aa or not wa: continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            zh = mlp_(build(out_p, h_p, dt, wa), 'z_post', 'gelu')[:32]
            ZH.append(zh.numpy()); ZT.append(Xt[r0_+k_, 960:].numpy()); ZTn.append(Xt[r0_+k_+1, 960:].numpy()); ZTp.append(Xt[r0_+k_-1, 960:].numpy())
            if len(ZH) >= max_n: break
        if len(ZH) >= max_n: break
    return [np.array(a) for a in (ZH, ZT, ZTn, ZTp)]
bld = lambda o,h,dt,wa: torch.cat([o + ln_(h.mean(0),'core.out_norm') + G_(dt), E_(wa), U0])
ZH, ZT, ZTn, ZTp = zp_collect(stays_t, bld)
for nm_, A in [('same', ZT), ('next', ZTn), ('prev', ZTp)]:
    print(nm_, 'rel err', round(float(np.linalg.norm(ZH-A)/np.linalg.norm(A)),3))
pc = [np.corrcoef(ZH[:,j], ZT[:,j])[0,1] for j in range(32)]
sl = [np.polyfit(ZH[:,j], ZT[:,j], 1)[0] for j in range(32)]
print('per-dim corr', np.round(pc, 2)); print('per-dim slope', np.round(sl, 2))
print('std mine', np.round(ZH.std(0)[:8],3), 'std true', np.round(ZT.std(0)[:8],3))
print('mean mine', np.round(ZH.mean(0)[:8],3), 'mean true', np.round(ZT.mean(0)[:8],3))
```