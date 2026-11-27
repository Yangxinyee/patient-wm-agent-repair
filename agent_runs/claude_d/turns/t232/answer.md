E3-ins works cleanly. The norepinephrine readouts are unchanged, as intended:

| Readout | Original | E3-ins | Target (linear estimate) |
|---|---|---|---|
| insulin x2 | −1.73 | −16.55 | −13.83 |
| insulin x1/2 | +0.91 | +7.86 | +6.92 |
| NE x2 | −0.08 | −0.05 | about +4.5 |
| NE x1/2 | −0.48 | −0.48 | about −2.3 |

Insulin now overshoots by about 15–20%. The x2 to x1/2 ratio is about −2.1, so the channel is close to linear. The model's own propagation therefore carries a working correction strongly enough, and C = −3.32 is a bit too much. On a linear rescaling of the edit's contribution, the factor would be about 0.82 to 0.87, giving C ≈ −2.8.

Before rescaling I will check three things on this pass:
1. The factual arm (recorded doses) of the edited model should track the original's at hour 13 and after. That confirms the bf16 fix on the GPU.
2. The hourly profile against the hourly targets. If it rises too fast, the per-hour mismatch tells me whether a single scale factor is enough.
3. The lag coefficients for x2 and x1/2 should agree, which would confirm linearity.

```python
d10e = pd.read_csv('/work/fwd_10/dose_edit_edited.csv'); d10o = pd.read_csv('/work/fwd_10/dose_edit_original.csv')
L10 = lagD(d10e)
d10e['tgt'] = np.where(d10e.edit.str.startswith('ins'), L10 @ beta_ins[:9], L10 @ bne[:9]); d10e['orig'] = d10o['shift'].values
d10e['fbias'] = d10e.factual.values - d10o.factual.values
for ed_ in ['insulin_x2','insulin_half']:
    mk_ = (d10e.edit==ed_) & d10e.reached; sub_ = d10e[mk_]
    byh = sub_.groupby('hour')[['tgt','orig','shift','fbias']].mean().round(1)
    print(ed_, 'mean tgt %.2f edit %.2f | factual-arm bias mean %.2f' % (sub_.tgt.mean(), sub_['shift'].mean(), sub_.fbias.mean()))
    print('  tgt', list(byh.tgt)); print('  edt', list(byh['shift'])); print('  fbias', list(byh.fbias))
    Lm = L10[mk_.values][:, 1:7]
    print('  lagcoef edit', np.round(np.linalg.lstsq(Lm, sub_['shift'].values, rcond=None)[0], 2), ' beta', np.round(beta_ins[1:7],2))
fb13 = d10e[(d10e.edit=='insulin_x2') & (d10e.hour==13)].fbias
print('h13 factual-arm bias pct10/50/90', np.round(np.percentile(fb13, [10,50,90]),2))
nb = d10e[(d10e.edit=='ne_x2') & d10e.reached].fbias
print('MAP factual-arm bias mean %.2f' % nb.mean())
```