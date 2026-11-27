The factual CSV has one row per reading, with the observed value, the model's predicted mean and the NLL. Now I will run the regression. For each finger reading at hour t whose previous hour t−1 had a finger-tier insulin dose, I will regress:
- the model's predicted mean at t on the dose at t−1, controlling for the reading at t−1 (cubic), tier dummies and earlier readings;
- the observed value on the same terms, for comparison.

A slope near −2 would mean the model encodes the causal effect; near 0, it does not.

```python
fa['h'] = fa.hour.astype(int)
pm = fa[fa.event=='glucose_finger'].set_index(['stay_id','h'])
Pt = Ptu.copy()
Pt['tf'] = tier(Pt.glucose_finger); Pt['D'] = Pt.insulin.fillna(0)
Pt['pm'] = pm.predicted_mean.reindex(Pt.index)
Pt['pm1'] = Pt.groupby(level=0).pm.shift(-1)
Pt['G1'] = Pt.groupby(level=0).glucose_finger.shift(-1)
Pt['gp1'] = Pt.groupby(level=0).glucose_finger.shift(1)
stays100 = tu_ids[:100]
q = Pt.loc[stays100]
q = q[q.glucose_finger.notna() & (q.tf>=1) & q.insulin.notna() & q.pm1.notna() & (q.index.get_level_values('h')>=0)]
q = q[~q.glucose_finger.round().isin([150,180,220,260])]
g = q.glucose_finger.values
gp = q.gp1.values; hasgp = ~np.isnan(gp)
X = [np.ones(len(q)), g, g**2/100, g**3/1e4] + [(q.tf.values==t).astype(float) for t in [2,3,4]] + [np.where(hasgp,gp,0), hasgp.astype(float), q.D.values]
nm = ['c','g','g2','g3','t2','t3','t4','gp','hasgp','D']
print("n=",len(q))
print("model predicted mean:", ols(q.pm1.values, X, nm).loc['D'].values)
print("observed next:", ols(q.G1.values, X, nm).loc['D'].values)
```