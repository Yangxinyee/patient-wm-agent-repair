With richer controls, norepinephrine shows a possible slow negative effect. The local projection is +0.14 at 1 hour but drifts negative over 3–12 hours, reaching about −0.6 ± 0.2 at 10 hours and −0.57 ± 0.25 at 12 hours (per unit innovation, which also raises later rates: 0.67, 0.48, 0.35, …).

If that is real, it implies an open-loop effect of roughly −0.1 to −0.2 mmHg per rate-hour, with the opposite sign to real medicine. Under x2 (about +0.5 mcg/kg/min sustained) that would give a readout of perhaps −0.3 mmHg, against the model's −0.03. The evidence is weak: the estimates at different horizons are correlated and k=1 is positive.

Next I will do the same richer-controls analysis for insulin, over horizons up to 12 and by base level, also checking glucose_lab. That tells me whether insulin hides a slow cumulative effect, or an effect that varies with glucose level and averages out in the pooled estimate. If insulin stays at zero at every horizon and level, I will focus the precision work on norepinephrine.

```python
for k in range(9,13):
    P[f'gf_f{k}'] = g['gf'].shift(-k); P[f'ins_f{k}'] = g['ins'].shift(-k)
P['gf_prev2'] = g['gf'].shift(2)
d = P[(P['ins']>0) & P['gf'].notna()].copy().reset_index()
d['eps'] = d['ins'] - d.groupby('gbin5')['ins'].transform('mean')
d['gp_f'] = d['gf_prev'].fillna(d['gf']); d['has_gp']=d['gf_prev'].notna().astype(float)
d['gp2_f'] = d['gf_prev2'].fillna(d['gp_f']); d['ins_prev_f']=d['ins_prev'].fillna(0)
ctrl = ['gp_f','has_gp','gp2_f','ins_prev_f']
LPi=[]; Ri=[1.0]
for k in range(1,13):
    d[f'dG{k}'] = d[f'gf_f{k}'] - d['gf']
    b,se,n = fe_ols(d, f'dG{k}', ['eps']+ctrl, 'gbin5')
    d[f'if{k}'] = d[f'ins_f{k}'].fillna(0)
    b2,se2,_ = fe_ols(d, f'if{k}', ['eps']+ctrl, 'gbin5')
    LPi.append((b[0],se[0])); Ri.append(b2[0])
    print(f"k={k}: LP {b[0]:+.3f} ({se[0]:.3f}) n={n} | R {b2[0]:+.3f} ({se2[0]:.3f})")
# glucose_lab outcome
for k in [1,2,4,8]:
    d[f'gl_f{k}'] = d.merge(P[['gl']].groupby(level=0).shift(-k).reset_index().rename(columns={'gl':'glf'}), on=['stay_id','ih'], how='left')['glf'].values
    d[f'dL{k}'] = d[f'gl_f{k}'] - d['gf']
    b,se,n = fe_ols(d, f'dL{k}', ['eps']+ctrl, 'gbin5')
    print(f"lab k={k}: {b[0]:+.3f} ({se[0]:.3f}) n={n}")
```