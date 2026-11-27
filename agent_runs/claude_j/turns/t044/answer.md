With only hours where insulin was actually given, the lag-1 slope is -1.93 ± 0.05 per unit/h, similar across bands (-1.86, -2.01, -1.95, -2.36). The earlier -1.0 came from about 8,800 readings above 150 with no insulin record, whose "deviation" isn't random noise. So the -2/unit kernel and the truth table stand.

The model-side regression (slope -0.01) used the same contaminated deviation, so I'll redo it on given-only rows of the factual pass. If the model's slope stays near 0 while the actual slope there is near -2, the model really ignores the exogenous dose. A slope near -2 would mean it learned the effect within support and fails only when extrapolating.

```python
Qg = Q.merge(insu.assign(h=insu.hr_read.astype(int))[['stay_id','h']].assign(given=True), on=['stay_id','h'], how='left')
Qg['given'] = Qg.given.fillna(False).astype(bool)
Qg = Qg[(Qg.g0>150)&(Qg.given)].copy()
print(len(Qg))
for col in ['g1','predicted_mean']:
    print(col, 'dev slope %.3f (se %.3f)' % reg(Qg[col].values, Qg))
# also: model prediction vs rate within band, controlling g0 linearly
for bb in [1,2,3]:
    c = Qg[Qg.base==bb]
    X = np.column_stack([np.ones(len(c)), c.g0, c.rate])
    b1,*_ = lstsq(X, c.predicted_mean.values, rcond=None); b2,*_ = lstsq(X, c.g1.values, rcond=None)
    print('band', bb, len(c), 'model slope %.3f actual slope %.3f' % (b1[2], b2[2]))
```