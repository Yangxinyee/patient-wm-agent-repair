For patients off norepinephrine the start rule is sharp: start if and only if MAP < 65 (46% at exactly 65, a rounding effect). There is no jump in MAP change at the threshold: +3.47 vs +3.63 mmHg at 1 h, +3.44 vs +3.69 at 2 h, +3.55 vs +3.51 at 3 h. So starting norepinephrine does nothing to MAP, just as starting insulin does nothing to glucose. Two separate natural experiments (randomised dose multipliers, and sharp start thresholds) now agree that **both drugs are inert in this ICU**.

Before I commit, I will check the parts of the hypothesis that matter for the readout:
- the insulin margin effect restricted to dose hours ≥ 12 (the readout window) at lags 1–12;
- the norepinephrine margin effect for hours ≥ 12;
- norepinephrine heterogeneity by MAP level.

All of these should come out about 0. After that I will use the forward tool to see what the original model currently predicts for the four dose edits.

```python
# insulin: random part of dose = ins - base*E[m|base]; use within-stratum residual
d2 = G[G.glucose_finger.notna() & (G.ins>0)].copy()
d2 = d2[~d2.glucose_finger.isin([150,180,220,260,300])]
d2['base'] = base_ins(d2.glucose_finger.values); d2 = d2[d2.base>0]
grp2 = G.groupby(level='stay_id')
for k in range(7,13): G[f'gf_p{k}'] = grp2.glucose_finger.shift(-k)
for k in [5]: G[f'gf_p{k}'] = grp2.glucose_finger.shift(-k)
d2 = d2.join(G[[f'gf_p{k}' for k in [5,7,8,9,10,11,12]]])
d2['hh'] = d2.index.get_level_values('h')
res = []
for late in [False, True]:
    dd0 = d2[d2.hh>=12] if late else d2
    st0 = dd0.index.get_level_values('stay_id').values
    for k in range(1,13):
        sel = dd0[f'gf_p{k}'].notna().values
        if sel.sum()<2000: continue
        dd = dd0[sel]; X = design(dd, [dd.ins.values]); y = (dd[f'gf_p{k}']-dd.glucose_finger).values
        b, se, _ = ols(y, X, st0[sel]); res.append((late, k, sel.sum(), round(b[-1],3), round(se[-1],3)))
print(pd.DataFrame(res, columns=['h>=12','lag','n','coef_per_unit','se']).to_string())
```