The panel works. Glucose_finger is present in about 52-59% of hours, rising over the stay. Glucose_lab is present in 20%, map_arterial in 46%, map_cuff in 61%. Insulin runs in 31-42% of hours and norepinephrine in about 32%. Insulin switches on and off within stays, so there is plenty of within-stay dose variation.

To identify causal effects I need to know how the clinician's dose depends on what is recorded. Next I will relate the insulin decision (any insulin, and its rate) to the glucose readings at the same hour, which are taken 5 minutes before the dose, and at earlier hours. I will do the same for norepinephrine and MAP. If dose is almost a deterministic function of the same-hour reading, the policy is mostly observed. If there is large residual variation, I need to check whether it is exogenous noise or reflects something the clinician sees that the record does not.

```python
P = P.sort_values(['stay_id','h']).reset_index(drop=True)
g = P.groupby('stay_id')
for c in ['GF','GL','MA','MC','I','N']:
    P[c+'_prev'] = g[c].shift(1)
# last observed GF (carry forward) before or at hour h
P['GF_ff'] = g['GF'].ffill()
P['MA_ff'] = g['MA'].ffill(); P['MC_ff']=g['MC'].ffill(); P['GL_ff']=g['GL'].ffill()
P['anyI'] = P.I.notna().astype(int); P['anyN']=P.N.notna().astype(int)
d = P[(P.h>=0)&P.GF.notna()]
bins = np.arange(60,400,20)
tab = d.groupby(pd.cut(d.GF,bins)).agg(n=('anyI','size'), pI=('anyI','mean'), meanI=('I','mean'), sdI=('I','std'))
print(tab.to_string())
d2 = P[(P.h>=0)&P.MA.notna()]
tab2 = d2.groupby(pd.cut(d2.MA,np.arange(20,140,5))).agg(n=('anyN','size'), pN=('anyN','mean'), meanN=('N','mean'), sdN=('N','std'))
print(tab2.to_string())
```