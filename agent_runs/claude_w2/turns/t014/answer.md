The NE policy is a titration with memory:
- When NE is off, it starts at MAP < 65 (65 is a 46% tie) at a low rate, about 0.05.
- When NE is on, it nearly always continues while MAP < 72, and continues 81-87% of the time even at high MAP.
- The rate when on is U-shaped in MAP: highest at low MAP, lowest around 72-75, and higher again above 80. The upturn above 80 suggests that patients on high rates have higher MAP, either because NE raises MAP or because the titration lags.

To use this for causal estimation I need the titration rule. That means rate_t as a function of rate_{t-1} and MAP_t (arterial versus cuff), plus whether there is residual noise that is independent of everything observed. Next I'll look at the ratio rate_t/rate_{t-1} by MAP band, its spread, its dependence on other covariates, and the stopping rule.

```python
v = w[(w.nor_prev>0)].copy()
v['ratio'] = v.nor0 / v.nor_prev
v['mb'] = pd.cut(v.map_now,[0,50,55,60,62,64,65,66,68,70,72,74,75,76,78,80,85,90,200])
print(v.groupby('mb', observed=True).ratio.describe().round(3).to_string())
vv = v[v.nor0>0]
print(vv.groupby('mb', observed=True).apply(lambda d: pd.Series({'mean_lr':np.log(d.ratio).mean(), 'sd_lr':np.log(d.ratio).std(), 'p_cap':(d.nor0>=1.0).mean()})).round(3).to_string())
# which MAP does the policy use when both present?
b = x[x.ma.notna() & x.mc.notna() & (x.nor_prev==0)]
b2 = b[(b.ma-65)*(b.mc-65) < 0]
print('both present, disagree around 65:', len(b2), 'start rate by which is <65:',
      b2.assign(ma_low=b2.ma<65).groupby('ma_low').nor.apply(lambda s: s.notna().mean()).to_dict())
```