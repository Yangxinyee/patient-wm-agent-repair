Two variants helped: no change in the gap scaling and age in hours (which helped glucose but hurt MAP). Glucose still has a 5–6 mg/dL MAE, so the register or type is probably still wrong.

Next I will scan all 15 type indices and regress the glucose prediction error on the register inputs (has-last, age, last z, dt). Strong structure in that regression shows which input is mis-encoded. If it shows none, the problem lies elsewhere, for example the state row used.

```python
for tidx in range(15):
    pm,_ = predict(R, type_idx=tidx, feat_mode='A', nz=8)
    print(tidx, mae_by_event(pm))
pm,_ = predict(R, **base)
err = pm - R.pm.values
D = R.copy(); D['err']=err; D['la_s']=np.log1p(D.age)/10; D['ageh']=D.age/3600
for ev in ['glucose_finger','map_cuff']:
    d = D[D.event==ev]
    print(ev, 'err mean by has:', d.groupby('has').err.agg(['mean','std','size']).round(2).to_dict())
    print('  err by age(h) bins:', d[d.has==1].groupby(pd.cut(d[d.has==1].ageh,[0,1.01,2.01,3.01,6,30])).err.agg(['mean','std','size']).round(2).to_dict())
    print('  corr err~zl', np.corrcoef(d.err, d.zl)[0,1].round(3), ' err~obs-pm(tool)', np.corrcoef(d.err, d.obs-d.pm)[0,1].round(3))
```