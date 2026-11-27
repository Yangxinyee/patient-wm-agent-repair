I had built a numpy reconstruction of the heads from the fwd_3 states: view_, q_, wc_ (with_code), p_i, nll_mine and mn16, compared against the fwd_2 factual predictions in F. I don't know how well it matched or what Xf and yy are for. Next I'll print the rest of the variable list and check the agreement: the mean absolute difference between my reconstructed means (mn_i, mn16, mn0) and the tool's predicted_mean, and between nll_mine and the tool's nll. A small difference (below about 0.5 mg/dL) would confirm I can compute head outputs from states offline. Edits to head weights could then be checked without forward passes.

```python
print(' | '.join(items)[3800:6000])
obs_pm = F['predicted_mean'].values
for nm in ['mn','mn_i','mn0','mn16']:
    v = globals()[nm]
    ok_ = np.isfinite(v)
    print(nm, 'MAD vs tool:', np.round(np.nanmean(np.abs(v[ok_]-obs_pm[ok_])),4), 'n', ok_.sum())
print('nll MAD', np.nanmean(np.abs(nll_mine - F['nll'].values)), 'mads', mads)
print(F.event.value_counts().to_dict())
```