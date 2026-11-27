The clinicians' feedback on later doses is small: a 1 unit/h deviation at t lowers the next doses by only 0.03–0.05 units. The finger-reading schedule barely responds; a reading at t+1 is certain whenever insulin ran at t. So the lag responses I measured are close to the fixed-dose impulse response. A unit/h of insulin at hour t lowers glucose by about 2.0, 2.3, 1.8, 1.3, 0.7 and 0.45 mg/dL over the next 6 hours. A sustained increase would accumulate to roughly -8 to -10 mg/dL per unit/h.

The x2, x0, x1.5 and x3 readouts all depend on whether this is linear in dose. Next I'll check linearity:
- the immediate slope separately within each glucose band, since typical rates differ (about 1, 2, 3, 4+ units/h) and scatter widths differ;
- the response by quantile of the deviation.

Equal slopes across bands and a straight-line response to the deviation would support a linear dose-response, so x2 is simply twice the recorded dose's effect. Slopes that grow or shrink with rate would mean I must model the curvature before computing the counterfactuals.

```python
for k in [1,2,3]:
    s = L.dropna(subset=['g0',f'g{k}'])
    out=[]
    for b in sorted(s.band.unique()):
        sb = s[s.band==b]
        r = ols((sb[f'g{k}']-sb.g0).values, [sb.dev.values, sb.g0.values], ['dev','g0'])
        out.append((b, len(sb), sb.ins.mean().round(2), sb.ins.std().round(2), r.loc['dev','b'].round(3), r.loc['dev','se'].round(3)))
    print("lag",k, out)
# nonparametric: in band 3 (180-220, mean rate 2), mean (g1-g0) by rate value
s = L.dropna(subset=['g0','g1']); s=s[s.band==3]
print(s.groupby(s.ins.clip(0,5)).apply(lambda x: pd.Series({'n':len(x),'dg':(x.g1-x.g0).mean(),'g0':x.g0.mean()})).round(2))
s = L.dropna(subset=['g0','g1']); s=s[s.band==4]
print(s.groupby(s.ins.clip(0,7)).apply(lambda x: pd.Series({'n':len(x),'dg':(x.g1-x.g0).mean(),'g0':x.g0.mean()})).round(2))
```