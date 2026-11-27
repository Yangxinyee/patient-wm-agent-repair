The insulin readout generalises well. On unseen tuning stays its correlation with the true dose deviation is 0.75 (R2 0.56), close to in-sample, so it isn't overfit. It responds about 1.33 per unit of my record-based deviation.

The cost is concentrated in two places: +0.0078 per glucose_finger row where insulin ran in the previous hour (981 rows), and +0.0035 on the 557 rows without insulin in the previous hour. On the latter the correction should be off but averages |0.52| mg/dL. The likely cause is that the previous bundle there holds the END atom of the infusion from two hours before, which turns the gate on.

If the leaked correction tracks that earlier dose's deviation, it acts as a useful lag-2 correction and I should keep it. If it is unrelated, removing it saves about 2 nats per 100 stays that could go to a larger insulin coefficient. I'll check the leaked shift against the deviation of the insulin dose at h-2 and against the previous bundle's makeup.

```python
g['d2'] = [insT.at[s,h-2] if (h-2) in insT.columns and pd.notna(insT.at[s,h-2]) else 0.0 for s,h in zip(g.stay_id, g.hour)]
g['tr2'] = [tier_val(gfG.at[s,h-2]) if (h-2) in gfG.columns else None for s,h in zip(g.stay_id, g.hour)]
g['dev2'] = [d - (Etab[int(t)] if not pd.isna(t) else 0.03) for d,t in zip(g.d2, g.tr2)]
off_ = ~on
m2 = off_ & (g.d2.values > 0)
m0 = off_ & (g.d2.values == 0)
print('no-ins-prev-hour rows: with insulin at h-2: %d, none at h-2: %d' % (m2.sum(), m0.sum()))
print('  mean|shift| with ins h-2 %.3f, none %.3f' % (np.abs(g.sh.values[m2]).mean(), np.abs(g.sh.values[m0]).mean()))
lk = g.sh.values[m2]/(-1.9)
print('  leak readout vs dev2: corr %.3f slope %.3f ; dNLL rows with ins h-2 %.4f, none %.4f' % (np.corrcoef(lk, g.dev2.values[m2])[0,1], np.cov(lk, g.dev2.values[m2])[0,1]/g.dev2.values[m2].var(), g.dnll.values[m2].mean(), g.dnll.values[m0].mean()))
```