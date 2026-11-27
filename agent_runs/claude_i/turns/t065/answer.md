Changing how the posterior context is built moves the match a lot. Dropping the time flow raises measurement bundles from 0.33 to 0.71, and dropping the gap raises drug bundles to 0.69. So my context formula is wrong somewhere, even though the head replica tolerated it (its NLL match could hide an error in a term the head barely uses).

Next, I will scan a grid of context formulas on all later bundles:
- time-flow units: seconds, minutes, hours or none;
- gap input: log of seconds/10, log of hours/10, log1p(seconds)/10, or no gap.

For each I record the posterior-mean correlation on measurement-only and drug-only bundles. The right formula should push both close to 1. If none does, the remaining error is in the encoders. Then I'll fix ctx at the best variant and search the encoder choices on drug-only bundles.

```python
res_=[]
gapopts = {'ls10': np.log(dtb)/10, 'lh10': np.log(dtb/3600)/10, 'l1p10': np.log1p(dtb)/10, 'none': None}
for fl_nm, fl_c in [('sec',1.0),('min',1/60),('hr',1/3600),('none',0.0)]:
    Hf = Hraw_*np.exp(-(dtb*fl_c)[:,None]*np.exp(P['fa'])[None,:])[:,None,:]
    base_ = X3[pkb,:192] + lnorm(Hf.mean(1), P['ow'], P['ob'])
    for g_nm, gin in gapopts.items():
        cx = base_ + (0 if gin is None else mlp2(gin[:,None], 'gap'))
        zo_ = mlp2(np.c_[cx, E0, U0], 'z_post')[:, :32]
        cm = np.corrcoef(zo_[kinds=='meas'].ravel(), tgt[kinds=='meas'].ravel())[0,1]
        cd = np.corrcoef(zo_[kinds=='drug'].ravel(), tgt[kinds=='drug'].ravel())[0,1]
        res_.append((fl_nm, g_nm, round(cm,4), round(cd,4)))
for r in sorted(res_, key=lambda r: -(r[2]+r[3])): print(r)
```