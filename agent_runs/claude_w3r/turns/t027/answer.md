The cross-fitted insulin policy explains 86% of rate variance (innovation SD 0.39), and the placebo on glucose 4 h earlier passes (−0.07 ± 0.05).

A remaining risk for the insulin response is selection. glucose_finger is observed only when measured, and if measurement depends on recent glucose (which insulin lowers), the reduced-form sample is post-treatment selected. Next I will look at what drives glucose_finger measurement: the last glucose value, the insulin status, and the hour. That tells me whether this bias matters and how to handle it.

```python
rows=[]
for t in range(2,T):
    rows.append(pd.DataFrame({'meas':~np.isnan(gf[:,t]),'gl':gf_last[:,t-1],'gage':gf_age[:,t-1],'ins_prev':ins_r[:,t-1]>0,
                              'lab':~np.isnan(A['glucose_lab'][:,t]),'t':hours[t]}))
M_ = pd.concat(rows).dropna(subset=['gl'])
M_['gb']=pd.cut(M_.gl,[0,100,140,150,160,180,200,250,400])
print(M_.groupby(['gb','ins_prev']).meas.agg(['size','mean']).round(3).unstack().to_string())
print(M_.groupby(M_.gage.clip(0,6)).meas.agg(['size','mean']).round(3).to_string())
```