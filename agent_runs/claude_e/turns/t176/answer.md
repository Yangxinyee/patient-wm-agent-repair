The error comes from pandas turning missing tiers (None) into NaN in the tier column, so the table lookup failed. I'll test for missing values with pd.isna and rerun the same out-of-sample check.

```python
g['dev'] = [d - (Etab[int(t)] if not pd.isna(t) else 0.03) for d,t in zip(g.d, g.tr)]
on = (g.d > 0).values
ru = g.sh.values[on]/(-1.9); dv = g.dev.values[on]
print('gf rows %d, with insulin prev hour %d' % (len(g), on.sum()))
c = np.corrcoef(ru, dv)[0,1]
print('out-of-sample: corr(readout, dev) %.3f  R2 %.3f  slope readout-on-dev %.3f  readout sd %.3f dev sd %.3f' % (c, c**2, np.cov(ru, dv)[0,1]/dv.var(), ru.std(), dv.std()))
print('rows without insulin: mean |shift| %.3f  (gate leakage)' % np.abs(g.sh.values[~on]).mean())
print('dNLL on gf: insulin rows %.4f (n %d), other rows %.4f (n %d)' % (g.dnll.values[on].mean(), on.sum(), g.dnll.values[~on].mean(), (~on).sum()))
print('WC' in globals() and np.shape(WC), 'RR' in globals() and RR.shape)
```