Run 188 found unused dimensions in head 1's value space and an almost silent carrier direction (natural projection rms 0.001). It also exposed a serious problem. The insulin-rate readout fitted only on insulin-present rows produces values up to 925 at NE-only bundles, and the value path is not gated, so that readout cannot be used. The jointly fitted readout (zero where insulin is absent) is the safe choice.

I'll check its mean bias and rms by dose multiplier and by bundle type (insulin present, NE-only, measurement), and do the same for the NE readout (rate minus 0.542). Small biases (≤0.02) and rms of about 0.15 or less would be acceptable, since errors mostly average out across bundles.

```python
BB3['tI'] = BB['tI']
grp_rows = np.concatenate([np.repeat(np.select([iI[VAR[k]['selr']]&~iN[VAR[k]['selr']], iI[VAR[k]['selr']]&iN[VAR[k]['selr']], ~iI[VAR[k]['selr']]&iN[VAR[k]['selr']]], ['ins','combo','ne'], 'none'),4) for k in VAR] + [np.array(['meas']*len(Fn))])
var_rows = np.concatenate([np.array([str(k)]*len(VAR[k]['F'])) for k in VAR] + [np.array(['meas']*len(Fn))])
for nm in ['tI','fN']:
    e_ = FF@BB3[nm] - TT[nm]
    s1 = ' '.join('%s:%+.3f/%.3f' % (k, e_[(var_rows==str(k)) & (TT['pI' if nm=='tI' else 'pN']>0.5)].mean(), np.sqrt((e_[(var_rows==str(k)) & (TT['pI' if nm=='tI' else 'pN']>0.5)]**2).mean())) for k in VAR)
    s2 = ' '.join('%s rms %.3f max %.3f' % (g, np.sqrt((e_[grp_rows==g]**2).mean()), np.abs(e_[grp_rows==g]).max()) for g in ['ins','combo','ne','meas'])
    print(nm, 'present bias/rms by variant:', s1); print('   by group:', s2)
```