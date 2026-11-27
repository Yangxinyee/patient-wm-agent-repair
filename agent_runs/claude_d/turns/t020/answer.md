The per-row layout is now clear. Each row is one recorded glucose_finger reading (or map_arterial reading for the norepinephrine edits) at hours 13–24, with:
- the model's mean shift over 20 paths and its standard error;
- the factual predicted mean;
- a `reached` flag that is true only after the first drug record at or after 12:05.

For stay 2020000000 the shift grows slowly, to about −1.6 by hour 23. For stay 2020000032, with heavy insulin, it is only about −2, while my data estimate implies about −2 mg/dL per u/h for every hour of extra insulin, accumulating.

To produce the true counterpart of this readout I need a structural model of glucose dynamics: how much of a dose's effect persists, whether the response is linear (x2 vs x1/2), and whether it varies by patient. Next I will:
- describe the glucose dynamics without insulin (drift and mean reversion);
- check whether the one-hour insulin effect varies with glucose level, hour or norepinephrine status;
- check for cross-effects (insulin → MAP, norepinephrine → glucose) using the same random-noise approach.

```python
# 1) dynamics without insulin: consecutive gf readings, no insulin at t
dz = c[c.gf.notna() & c.gf_p1.notna() & (c.ins0==0)]
print('no-ins: n', len(dz)); 
print(dz.groupby(pd.cut(dz.gf,[0,80,100,120,130,140,150,160,180,200,250]),observed=True).apply(lambda x: pd.Series({'n':len(x),'d1':(x.gf_p1-x.gf).mean(),'sd':(x.gf_p1-x.gf).std()})).round(2).to_string())
# 2) one-hour insulin effect heterogeneity: regress gf_p1 on gf, ins within groups
d = c[(c.gf.notna()) & (c.base_f>0) & c.ins.notna() & c.gf_p1.notna()].copy()
d['hr']=d.index.get_level_values('h')
def fit(s):
    Xs = pd.get_dummies(s.base_f.astype(int), prefix='b').astype(float)
    for col in list(Xs.columns): Xs[col+'_gf'] = Xs[col]*s.gf
    Xs['rate']=s.ins
    return LinearRegression().fit(Xs, s.gf_p1).coef_[-1]
for nm_, grp in [('nor on', d.nor0>0), ('nor off', d.nor0==0), ('hr<8', d.hr<8), ('hr 8-16', (d.hr>=8)&(d.hr<16)), ('hr>=16', d.hr>=16)]:
    print(nm_, grp.sum(), round(fit(d[grp]),3))
# 3) cross effects: insulin noise -> ma_p1 ; NE noise -> gf_p1
d2 = d[d.ma.notna() & d.ma_p1.notna()]
Xs = np.c_[d2.ma, d2.gf, d2.base_f, d2.ins]; print('ins->ma_p1 coef', np.round(LinearRegression().fit(Xs, d2.ma_p1).coef_,3))
dn2 = dn[dn.gf.notna() & dn.gf_p1.notna()]
Xs = np.c_[dn2.gf, dn2.ins0, dn2.ma, dn2.nor_prev, dn2.nor]; print('nor->gf_p1 coef', np.round(LinearRegression().fit(Xs, dn2.gf_p1).coef_,3))
```