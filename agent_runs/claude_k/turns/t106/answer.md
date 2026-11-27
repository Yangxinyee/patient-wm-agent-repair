The tool's per-reading file has the same 5,521 readings, with the observed value, predicted mean and fine-bin NLL. The bin centres are stored per reading code.

I'll join the tool's rows to mine on stay, hour and event. Then I'll compare (a) NLL, by correlation and mean difference, and (b) the predicted mean, from my probabilities and the per-code centres, against the tool's. Correlations of 0.99 or more with a small offset mean the recon is usable for costing edits. A correlation near 0.9 points to a structural gap, most likely the z prior draws, which I'd then need to model.

```python
R = R99.reset_index(drop=True).copy()
R['nll_rec'] = base_nll
mean_rec = np.zeros(len(R))
for ev in R['event'].unique():
    m = (R['event'] == ev).values
    c = np.array(vc[ev], dtype=float)
    mean_rec[m] = (p99[m] * c[None, :]).sum(1) if c.shape[0] == 17 else (p99[m][:, 1:] * c[None, :]).sum(1) / p99[m][:, 1:].sum(1)
R['mean_rec'] = mean_rec
M = R.merge(f4, on=['stay_id', 'hour', 'event'], how='inner')
print(len(M), 'dup keys', R.duplicated(['stay_id','hour','event']).sum())
for ev, g in M.groupby('event'):
    print(ev, len(g), 'nll tool %.4f rec %.4f corr %.3f' % (g.nll.mean(), g.nll_rec.mean(), np.corrcoef(g.nll, g.nll_rec)[0,1]),
          '| mean MAE %.2f corr %.4f' % (np.abs(g.predicted_mean - g.mean_rec).mean(), np.corrcoef(g.predicted_mean, g.mean_rec)[0,1]))
```