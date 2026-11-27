Eleven hidden units of the value head never fire on any of the 5,521 readings. Five of them (units 24, 26, 35, 125, 180) stay below −1 everywhere, so they have a clear safety margin. These are free capacity: I can turn pairs of them into a gated linear pathway without changing any current prediction.

Each pair would work like this:
- One unit gets gate + signal and the other gets gate alone, with opposite output weights.
- When the gate is open, the logits shift by (signal × an output vector of my choice).
- When the gate is closed, both units stay at zero.
- The gate opens only for glucose codes on the insulin pair, and only for MAP codes on the norepinephrine pair.

The open question is the signal. I need a direction in q that tracks the recent insulin dose beyond what the glucose readings already imply, i.e. the random part of the dose. That is the part the model underweights, and correcting it should leave the factual fit unbiased.

This run measures how well q encodes that part. For glucose readings it builds two targets: the previous hour's dose, and that dose minus its policy-expected value given the last glucose reading and whether insulin was on before. It then fits stay-grouped ridge regressions on q.

A clearly positive cross-validated R² for the residual part means a usable dose direction exists. An R² near zero means q barely carries dose information, and I would need to strengthen the dose encoding upstream.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
i0 = sidx[1010001440]; print('check ins_all row vs records:', ins_all[i0, :4], tr[(tr.stay_id == 1010001440) & (tr.event == 'insulin')].rate.values[:4])
gm = R99.event.isin(['glucose_finger', 'glucose_lab']).values & (R99.hour.values >= 1) & (R99.hour.values <= 24)
Rg = R99[gm].copy(); Qg = Q99[gm]
rowi = Rg.stay_id.map(sidx).values; th = Rg.hour.values.astype(int)
d1 = ins_all[rowi, th - 1]; d2 = np.where(th >= 2, ins_all[rowi, np.maximum(th - 2, 0)], 0)
lastg = np.array([glast[r_, t_ - 1] if True else 0 for r_, t_ in zip(rowi, th)])
print('glast sample', lastg[:5], 'gf at t-1', gf[rowi[:5], th[:5] - 1])
on_prev = (d2 > 0).astype(float)
bands = np.digitize(lastg, [130, 140, 150, 160, 170, 180, 200, 220, 240, 260, 300])
Z = np.column_stack([np.eye(12)[bands], np.eye(12)[bands] * on_prev[:, None]])
coef, *_ = np.linalg.lstsq(Z, d1, rcond=None); dres = d1 - Z @ coef
print('dose d1 mean %.2f sd %.2f; resid sd %.2f' % (d1.mean(), d1.std(), dres.std()))
grp = Rg.stay_id.values; gkf = GroupKFold(5)
for nm_, y_ in [('d1', d1), ('d1 resid', dres), ('d2', d2)]:
    yp = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), Qg, y_, groups=grp, cv=gkf)
    print(nm_, 'CV R2 %.3f' % (1 - np.mean((y_ - yp) ** 2) / np.var(y_)))
```