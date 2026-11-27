On tu1, the edit costs **+0.0037 per reading** at a = −2.5 (1.61959 → 1.62328):
- glucose_finger: +0.0108 per row (14.1 nats);
- lab glucose: +0.0056 per row (2.8 nats);
- MAP rows: about 0.

So far at −2.5: f3 0.0028, tu1 0.0037, and tu0 an estimated 0.0039 (it measured 0.0043 at −2.6). That is a mean of about 0.0035 with a spread of about ±0.0005, leaving a reasonable margin under 0.005.

Lab-glucose costs vary a lot between batches: +0.0005 per row on f3, +0.0056 on tu1, +0.0061 on tu0 at −2.6. Before spending passes on the last batches, I want to see where the cost sits. I'll pool the three tool factual files (fwd_13–15) and group the NLL change by the size and sign of the edit's predicted-mean shift, for finger and lab rows.

If a few large negative shifts carry much of the cost, clipping very negative readout values would help. It would barely touch the ×2 response, which uses positive deviations, and would have only a small effect on ×½. If the cost is spread evenly, there is nothing cheap to gain and I keep −2.5.

```python
FR = []
for k_ in [13, 14, 15]:
    fe_ = pd.read_csv(f'/work/fwd_{k_}/factual_edited.csv'); fo_ = pd.read_csv(f'/work/fwd_{k_}/factual_original.csv')
    d_ = fo_[['stay_id','hour','event','observed']].copy()
    d_['dpm'] = fe_.predicted_mean.values - fo_.predicted_mean.values; d_['dn'] = fe_.nll.values - fo_.nll.values; d_['k'] = k_
    FR.append(d_)
FR = pd.concat(FR, ignore_index=True)
for ev in ['glucose_finger','glucose_lab']:
    s_ = FR[FR.event==ev].copy()
    s_['bin'] = pd.cut(s_.dpm, [-100, -8, -5, -3, -1.5, -0.3, 0.3, 1.5, 3, 5, 8, 100])
    g_ = s_.groupby('bin', observed=True).dn.agg(['size','mean','sum']).round(4)
    print(ev, 'rows', len(s_), 'total dn %.1f' % s_.dn.sum(), '| by fwd:', s_.groupby('k').dn.sum().round(1).to_dict())
    print(g_.T.to_string())
```