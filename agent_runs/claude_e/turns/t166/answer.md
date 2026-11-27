The gate works exactly: on map_arterial rows the pair output equals the readout to within 0.001 in all 16 draws, and elsewhere it is exactly 0.

One problem remains. The unit-gain readout has mean 0.77 on target rows, while the true rate averages 0.45. An uncentred correction would shift every MAP-arterial prediction, including stays with no NE, and the original model is presumably unbiased there.

To decide how to centre and gate the NE correction, I'll look at the original model's MAP-arterial residual by recorded NE rate (0, low, mid, high). I'll also check how well an "NE in previous bundle" gate separates rows. If the residual is about 0 at rate 0 and rises with rate, the correction should be zero at rate 0, ideally gated to rows where NE is running.

```python
r_tm = res[tm7]; rate_tm = y_ne[tm7]; hn_tm = A_.hn.values[tm7]
bins_ = pd.cut(rate_tm, [-0.01, 0.0001, 0.3, 0.6, 1.01], labels=['0','<.3','.3-.6','.6-1'])
df_ = pd.DataFrame({'res': r_tm, 'bin': bins_, 'hn': hn_tm, 'ru': R.mean(0)[tm7]})
print(df_.groupby('bin', observed=True).agg(n=('res','size'), res=('res','mean'), ru=('ru','mean'), hn=('hn','mean')).round(3).to_string())
print('frac NE-running among map_arterial rows: %.3f' % (rate_tm>0).mean(), ' hn==1 frac %.3f' % (hn_tm==1).mean())
on = rate_tm > 0
ru_on = R.mean(0)[tm7][on]
cslope = np.cov(ru_on, r_tm[on])[0,1]/ru_on.var()
print('within NE-running rows: resid-opt slope on readout %.2f, intercept-at-mean %.2f, readout mean %.3f, rate mean %.3f' % (cslope, r_tm[on].mean(), ru_on.mean(), rate_tm[on].mean()))
```