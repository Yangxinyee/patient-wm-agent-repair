The 1-hour insulin effect is clean and linear: about −2.0 mg/dL on the next finger glucose per extra U/h. It is the same in every band from 150 to 300 mg/dL, so it neither scales with glucose level nor saturates. Each 0.5 U/h step in the within-band means lowers next-hour glucose by about 1 mg/dL. That is the immediate effect only. What the evaluation needs is how the effect accumulates over 1-12 hours with the other doses held fixed, which depends on persistence and mean reversion of glucose and on any delayed insulin action.

Next I'll measure:
- the total effect of the random part of insulin at t on finger glucose at t+1..t+6;
- the finger-lab relationship at the same hour (offset and noise);
- the lab measurement schedule, which matters for possible selection bias, since finger readings get sparser once glucose falls below 150.

```python
gg['ins_base'] = gg.groupby('band').ins0.transform('mean')
gg.loc[gg.ins.isna(),'ins_base'] = 0
# expected rate given band computed among rows with insulin
bm = gg[gg.ins.notna()].groupby('band').ins.mean()
gg['e_ins'] = np.where(gg.ins.notna(), gg.ins0 - gg.band.map(bm), 0.0)
both = gg[gg.gf.notna() & gg.gl.notna()]
print('gf-gl same hour: mean', (both.gf-both.gl).mean().round(2), 'sd', (both.gf-both.gl).std().round(2), 'n', len(both))
print('gl hours per stay', gg.groupby('stay_id').gl.count().describe().round(2).to_dict())
print('gl measured frac by band', gg.groupby('band').gl.apply(lambda s: s.notna().mean()).round(3).to_dict())
print('gf measured frac by band', gg.groupby('band').gf.apply(lambda s: s.notna().mean()).round(3).to_dict())
dd = gg[gg.ins.notna() & gg.gf.notna() & (gg.gf>=185)].copy()
for k in range(1,7):
    col = f'gf_l{k}'; s = dd[dd[col].notna()]
    lr = LinearRegression().fit(np.c_[s.e_ins, s.gf, s.h], s[col]-s.gf)
    s2 = dd[dd[f'gl_l{k}'].notna()]
    lr2 = LinearRegression().fit(np.c_[s2.e_ins, s2.gf, s2.h], s2[f'gl_l{k}']-s2.gf)
    print(k, 'gf: n', len(s), 'eff', round(lr.coef_[0],3), '| gl: n', len(s2), 'eff', round(lr2.coef_[0],3))
```