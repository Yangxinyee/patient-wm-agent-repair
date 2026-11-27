Having a partner reading in the same bundle makes no difference, so within-bundle conditioning is ruled out.

The gap appears when the previous bundle contained NE. For arterial MAP the excess NLL is 0.137 after an NE bundle versus −0.02 otherwise, and cuff MAP shows the same pattern. Finger glucose after insulin bundles matches.

So for MAP predictions after NE bundles, the real model uses something my head lacks. It may involve the running infusion. The doc's "pending" features (`pending_mlp`, `filter_gate`) feed the action heads, but the evidence suggests something similar reaches the MAP predictions.

This matters for the edit, because such a term could be where the model's NE effect lives. Next I check whether my MAP prediction error (mine minus tool) depends on:
- the NE rate just given;
- the previous rate;
- the change between them;
- whether NE stopped.

A strong dependence on the rate would mean a dose-dependent term reaches the MAP head directly.

```python
gsub = gg[['stay_id','h','nor','ins','nor_prev']].copy()
gsub['h1'] = gsub.h + 1
fa4['hi'] = fa4.hour.round().astype(int)
q4 = fa4.merge(gsub.rename(columns={'nor':'nor_lag1','ins':'ins_lag1','nor_prev':'nor_lag2'})[['stay_id','h1','nor_lag1','ins_lag1','nor_lag2']], left_on=['stay_id','hi'], right_on=['stay_id','h1'], how='left')
q4['err_pm'] = pm_[np.argsort(np.argsort(np.arange(len(fa4))))] - q4.predicted_mean if False else (fa4['nl_mine']*0 + pm_) .values - q4.predicted_mean.values
for ev in ['map_arterial','map_cuff']:
    d_ = q4[(q4.event==ev)]
    dn = d_[d_.ne_prev]
    print(ev, 'n', len(dn), 'corr err_pm with nor_lag1', round(np.corrcoef(dn.err_pm, dn.nor_lag1.fillna(0))[0,1],3),
          'with nor change', round(np.corrcoef(dn.err_pm, (dn.nor_lag1.fillna(0)-dn.nor_lag2.fillna(0)))[0,1],3),
          'dnl corr with nor_lag1', round(np.corrcoef(dn.dnl, dn.nor_lag1.fillna(0))[0,1],3))
    print(dn.groupby(pd.cut(dn.nor_lag1.fillna(0), [-0.01, 0.001, 0.2, 0.5, 0.8, 1.01])).err_pm.agg(['mean','std','size']).round(2).T)
    print('err_pm when NE stopped (nor_lag1 nan but ne_prev):', d_[d_.ne_prev & d_.nor_lag1.isna()].err_pm.describe().round(2).to_dict())
```