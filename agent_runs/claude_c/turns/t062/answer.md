The projection is flat. In the original model, the state change across an insulin bundle along the fitted innovation direction barely depends on rate in any tier (out about −2.7 to −3.1, h about −300 to −330). The state hardly encodes the dose at all.

From the paired rollout shifts, I reconstructed the model's lag-1 dose-response g(d):
- In the original, the slope is about −0.3 to −0.5 per U/h between 1 and 4 U/h and flat above about 5 U/h (truth −1.98, linear).
- With z ×5, the curve is compressed. The slope is about −1 to −1.6 per U/h between 1 and 3 U/h and flat above.

So input scaling only squeezes a saturating curve. It cannot produce a linear effect up to 8–10 U/h, and the best reachable insulin x2 readout this way is roughly −3 to −4 instead of −13.

One untested channel remains: the 31 z dimensions that barely vary at insulin bundles, with variance around 0.006. The earlier fit predicted rate from all 32 z dims with R² 0.78, so some low-variance dimension may encode the rate linearly at a small scale. Scaling `z_in` strongly along such a dimension would then give a linear dose channel, but only if that dimension is also quiet at reading and norepinephrine bundles.

Next I rank z dimensions by how well they track the rate at insulin bundles within tier, and by their variance elsewhere. A dimension with high within-tier correlation and low variance at other bundles is a candidate lever. If none exists, z offers nothing linear.

```python
xo = So['x']; Zall = xo[:, 960:992]
kinds = np.where(hi3, 'ins', np.where(hn3, 'ne_only', np.where(np.round(hr3-np.floor(hr3),4)==0, 'read', 'other')))
zi_ = Zall[ins_b][ok]; rate_ok = rate[ok]; tier_ok = tk[ok]
rows_out = []
for j in range(32):
    zj = zi_[:, j]
    # within-tier correlation with rate (remove tier means)
    zr = zj - pd.Series(zj).groupby(tier_ok).transform('mean').values
    rr_ = rate_ok - pd.Series(rate_ok).groupby(tier_ok).transform('mean').values
    cor = np.corrcoef(zr, rr_)[0,1] if zr.std()>0 else np.nan
    slope = np.polyfit(rr_, zr, 1)[0]
    rows_out.append(dict(j=j, sd_ins=zj.std(), cor_rate_within=cor, slope=slope,
                         sd_read=Zall[kinds=='read', j].std(), sd_ne=Zall[kinds=='ne_only', j].std(),
                         mean_ins=zj.mean(), mean_read=Zall[kinds=='read', j].mean()))
zt = pd.DataFrame(rows_out).set_index('j')
zt['sel'] = (zt.slope.abs()*0.58) / zt.sd_read.clip(lower=1e-6)
print(zt.sort_values('cor_rate_within', key=lambda s: -s.abs()).round(4).head(12).to_string())
```