`beta_ins` is indexed from lag 0 (zero) through lag 12, with peak effects at lags 1–3 (−2.0, −2.4, −2.1 mg/dL per u/h). Next I'll compute the true per-hour shift on sel1 for insulin x2, as the sum over lags of β times the dose change. I'll compare it hour by hour with the model's shift, and look at the hour-13 shift relative to the hour-12 dose. That last quantity is the model's immediate response, before any feedback. If the hour-13 ratio is already about 1/8 of −1.99, the dose-to-state pathway itself is weak. If it is near −2 and only later hours fall short, the problem is propagation.

```python
b=np.array(beta_ins)
d=de1[de1.edit=='insulin_x2'].copy()
def true_sh(sid,h,fac):
    i=sid2i[sid]; x=INS_U[i]; x=np.nan_to_num(x)
    return sum(b[k]*fac*x[h-k] for k in range(1,13) if h-k>=12)
d['true']=[true_sh(s,h,1.0) for s,h in zip(d.stay_id,d.hour)]
r=d[d.reached]
print(r.groupby('hour')[['shift','true']].mean().round(2).T.to_string())
print('overall', r[['shift','true']].mean().round(3).to_dict())
d13=d[(d.hour==13)]; x12=np.array([np.nan_to_num(INS_U[sid2i[s]][12]) for s in d13.stay_id]); m_=x12>0
print('h13 n',m_.sum(),'model/dose12',(d13['shift'].values[m_]/x12[m_]).mean().round(3),'median',np.median(d13['shift'].values[m_]/x12[m_]).round(3))
```