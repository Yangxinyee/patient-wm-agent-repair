The insulin response curve, in mg/dL per U/h for a one-hour dose, is precise:

| Lag (h) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10–12 |
|---|---|---|---|---|---|---|---|---|---|---|
| Effect | −2.04 | −2.30 | −1.83 | −1.34 | −0.76 | −0.51 | −0.26 | −0.19 | −0.06 | ≈ 0 |

The cumulative effect is about −9.2 mg/dL per U/h. The extra controls barely moved the estimates, which again supports exogeneity. A sustained dose therefore has a steady-state effect of about −9 mg/dL per U/h, reached within about 6–8 hours.

Next I will turn to norepinephrine and MAP, so the full set of true effects comes from the records before I spend forward passes. The NE policy looks like titration, with a rate that depends on the previous rate as well as current MAP. I need to find its exogenous component, as I did for insulin.

```python
P['map_a'] = P.map_arterial; P['map_c']=P.map_cuff
P['NE'] = P.norepinephrine
P['NEp'] = P.groupby(level=0).NE.shift(1)
q = P[P.index.get_level_values('h')>=0].copy()
q['has_a']=q.map_a.notna(); q['has_c']=q.map_c.notna()
print(pd.crosstab([q.has_a,q.has_c], q.NE.notna()))
# NE given vs MAP (arterial if available else cuff) and previous NE
q['m'] = q.map_a.fillna(q.map_c)
q['mbin'] = pd.cut(q.m, [0,50,55,60,62,64,65,66,68,70,75,80,90,200])
print(q.groupby([q.NEp.notna(), 'mbin']).agg(n=('m','size'), p=('NE', lambda x: x.notna().mean()), r=('NE','mean')).round(3).to_string())
# Ratio NE/NEp distribution when both present
both = q[q.NE.notna() & q.NEp.notna()]
print((both.NE/both.NEp).describe())
print((both.NE - both.NEp).round(2).value_counts().head(20))
```