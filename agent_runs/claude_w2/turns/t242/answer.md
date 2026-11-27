`M2` is the train teacher-forced residual table (94,387 readings, with the model's predicted mean `pm`), and it has the dose and policy lag columns I need. I'll regress the factual residual (value − pm) on lagged insulin deviations e and policy values p for finger glucose. Then I'll do the same with norepinephrine innovations u and expected NE x for arterial MAP. This is a falsifiable test:

- If the deviation coefficients are large and negative while the policy coefficients are about zero or the opposite sign, the model has absorbed the policy part. The edit should then act on deviations.
- If the policy coefficients match the deviation coefficients, the model is missing the absolute-dose effect.

```python
cols=set(M2.columns); print([c for c in ['d1','p1','e1','n1','x1','u1','res'] if c in cols])
if 'res' not in cols: M2['res']=M2.value-M2.pm
def show(tab): 
    t,n,s=tab; print('n',n,'sd',round(s,2)); print(t.T.to_string())
gfm=M2[M2.code==CODE['glucose_finger']]; mam=M2[M2.code==CODE['map_arterial']]
show(ols_tab(gfm,'res',['e1','e2','e3','e4','p1','p2','p3','p4']))
show(ols_tab(gfm,'res',['d1','d2','d3','d4','p1','p2','p3','p4']))
show(ols_tab(mam,'res',['u1','u2','u3','u4','x1','x2','x3','x4']))
show(ols_tab(mam,'res',['n1','n2','n3','n4']))
```