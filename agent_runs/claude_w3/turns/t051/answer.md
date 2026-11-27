The factual file has, for each reading, the observed value, the model's predicted mean and the NLL. Next I'll compare the model and the data on the same rows: insulin hours with a fresh policy glucose above 150 and a finger reading at the next hour.

I'll regress the change from the policy glucose to hour t+1 on band dummies plus the jitter part of the rate, j = rate − 1.056·base. The data coefficient on j should be near −2.1. If the model's coefficient is near zero, the model ignores the dose beyond its band. If it is near −2.1, the model learned the effect, and the weak rollout response has another cause.

```python
tu_ = tu.copy()
tu_['ih'] = np.floor(tu_['hour']+1e-6).astype(int)
gl = tu_[tu_.event.isin(['glucose_finger','glucose_lab'])]
pol = gl.assign(pr=(gl.event=='glucose_finger').astype(int)).sort_values('pr').groupby(['stay_id','ih']).value.last()  # finger preferred
insr_ = tu_[tu_.event=='insulin'].groupby(['stay_id','ih']).rate.sum()
fin = f2[f2.event=='glucose_finger'].copy(); fin['ih']=fin.hour.round().astype(int)
fin['G0'] = [pol.get((s_,h_-1),np.nan) for s_,h_ in zip(fin.stay_id,fin.ih)]
fin['r0'] = [insr_.get((s_,h_-1),0.0) for s_,h_ in zip(fin.stay_id,fin.ih)]
fin = fin[(fin.G0>150)&(fin.r0>0)].copy()
fin['base'] = np.select([fin.G0<=180,fin.G0<=220,fin.G0<=260,fin.G0<=300],[1,2,3,4],6)
fin['j'] = fin.r0-1.056*fin.base
Xd = pd.get_dummies(fin.base).astype(float).values
for yname in ['predicted_mean','observed']:
    y_ = (fin[yname]-fin.G0).values
    X_ = np.column_stack([Xd, fin.j.values, fin.G0.values/100])
    b_ = np.linalg.lstsq(X_, y_, rcond=None)[0]; r_ = y_-X_@b_
    se_j = np.sqrt(np.sum(r_**2)/(len(y_)-X_.shape[1]) * np.linalg.inv(X_.T@X_)[-2,-2])
    print(yname, 'n',len(y_), 'coef_j %.2f se %.2f'%(b_[-2],se_j), 'band means', np.round(b_[:Xd.shape[1]],1))
```