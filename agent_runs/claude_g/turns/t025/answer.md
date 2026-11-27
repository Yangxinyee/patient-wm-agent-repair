There is no discontinuity at any insulin threshold. At each of 150, 180, 220, 260 and 300 mg/dL the rate jumps by about 1 u/h (2 u/h at 300), yet the jump in later glucose change is within about ±0.9 mg/dL at 1–6 hours. The only larger value, −1.8 at 300/k=1, comes from a thin cell and reverses at k=2. So the base dose, the start and the random margin all have no effect: **insulin is inert**. The original model already says so (about ±0.1 mg/dL).

The one loose end is the small positive drift in the lag coefficients (+0.05 to +0.3 mg/dL per u/h, a few at about 2 SE in the late window). Taken literally it would imply a true insulin x2 readout of maybe +0.5 mg/dL rather than 0. I suspect a mild bias: m correlates about 0.015 with the previous glucose reading, so the multiplier may depend weakly on the glucose trend.

Test: rerun the lag regressions with richer pre-dose controls (G at h−1 and h−2, insulin at h−1 and h−2, lab glucose, hour). If the coefficients shrink toward 0, the drift was confounding and the truth is 0. If they stay positive, insulin may slightly raise glucose and I will quantify it.

```python
G['gf_m2'] = G.groupby(level='stay_id').glucose_finger.shift(2)
G['ins_m2'] = G.groupby(level='stay_id').ins.shift(2)
d3 = G[G.glucose_finger.notna() & (G.ins>0)].copy()
d3 = d3[~d3.glucose_finger.isin([150,180,220,260,300])]
d3['base'] = base_ins(d3.glucose_finger.values); d3 = d3[d3.base>0]
d3['hh'] = d3.index.get_level_values('h'); d3['m']=d3.ins/d3.base
# partial corr of log m with gf_m1 given G_h, base
x = d3.dropna(subset=['gf_m1'])
Xc = design(x); b,_,r_lm = ols(np.log(x.m.values), Xc); _,_,r_g1 = ols(x.gf_m1.values, Xc)
print('partial corr(log m, G_{h-1} | G_h, base):', np.corrcoef(r_lm, r_g1)[0,1], len(x))
def design2(dd):
    base_cols = design(dd)
    g1 = dd.gf_m1.values; has1 = ~np.isnan(g1); g1 = np.where(has1, (g1-200)/50, 0)
    g2 = dd.gf_m2.values; has2 = ~np.isnan(g2); g2 = np.where(has2, (g2-200)/50, 0)
    gl = dd.glucose_lab.values; hasl = ~np.isnan(gl); gl = np.where(hasl, (gl-200)/50, 0)
    return np.column_stack([base_cols, g1, has1, g2, has2, gl, hasl, dd.ins_m2.fillna(0).values, dd.hh.values/24])
res=[]
for late in [False, True]:
    dd0 = d3[d3.hh>=12] if late else d3
    st0 = dd0.index.get_level_values('stay_id').values
    for k in [1,2,3,4,6,8,10]:
        sel = dd0[f'gf_p{k}'].notna().values if f'gf_p{k}' in dd0 else None
        dd = dd0[sel]; X = np.column_stack([design2(dd), dd.ins.values]); y = (dd[f'gf_p{k}']-dd.glucose_finger).values
        b, se, _ = ols(y, X, st0[sel]); res.append((late, k, sel.sum(), round(b[-1],3), round(se[-1],3)))
print(pd.DataFrame(res, columns=['h>=12','lag','n','coef','se']).to_string())
```