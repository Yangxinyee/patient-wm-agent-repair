Finger and lab glucose measure the same thing: paired readings differ by -0.02 on average with SD 9 and correlation 0.974. Glucose_lab should therefore carry the same drug effect as glucose_finger. Arterial MAP runs 3 mmHg below cuff MAP, with SD 6.7.

The first regressions give Δglucose_finger ≈ -2.1 to -2.25 mg/dL per unit/h of insulin, with or without the current glucose as a control. That is a candidate effect but still possibly confounded. The fuller regression failed only because `Rv.ne` clashes with the pandas method `.ne`.

I'll rerun it with column indexing, adding flexible glucose-band controls, the previous rate and norepinephrine. If the coefficient stays near -2 under these controls, that supports a real effect of roughly 2 mg/dL per unit in the first hour. A large shift would mean confounding through the band structure, and I'd move to within-band (natural-experiment) variation only.

```python
Rv = Rv.rename(columns={'ne':'nor'}); R = R.rename(columns={'ne':'nor'})
gb = pd.get_dummies(pd.cut(Rv.g0,[0,100,120,140,150,160,170,180,190,200,210,220,240,260,300,1000]), drop_first=True).values.astype(float)
res = ols(Rv.dg.values, [Rv.ins.values, Rv.g0.values, Rv.ins_prev.values, Rv['nor'].values]+[gb[:,i] for i in range(gb.shape[1])],
          ['ins','g0','ins_prev','nor']+[f'gb{i}' for i in range(gb.shape[1])])
print(res.iloc[:5])
# within-band variation only: residualize ins on band-of-glast dummies x prev-on, use as instrument-like regressor
Rv['band'] = pd.cut(Rv.glast,[0,140,160,180,220,260,1e9], labels=False)
Rv['cell'] = Rv.band.astype(str) + '_' + (Rv.ins_prev>0).astype(str)
Rv['ins_dev'] = Rv.ins - Rv.groupby('cell').ins.transform('mean')
on2 = Rv[Rv.ins>0].copy()
on2['ins_dev'] = on2.ins - on2.groupby('band').ins.transform('mean')
print("on-insulin, within band dev:")
print(ols(on2.dg.values, [on2.ins_dev.values, on2.g0.values]+[ (on2.band==b).values.astype(float) for b in sorted(on2.band.unique())[1:]], ['ins_dev','g0']+[f'b{b}' for b in sorted(on2.band.unique())[1:]]).iloc[:3])
```