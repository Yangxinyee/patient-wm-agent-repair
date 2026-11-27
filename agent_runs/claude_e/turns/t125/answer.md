ED is not a rescaled `make_edit(1,1)`. In `make_edit`, the input weights and biases of the four units are fixed. Only the output columns and the transition-table entries depend on the coefficients, roughly linearly. ED was therefore built with some other (a_ins, a_ne), which I cannot read off reliably.

Rather than reverse-engineering ED, I will measure what an edit does. I will run the local head (`head_run`) with `make_edit(1,1)` on one training state batch. Then I compare its predicted means with the original's (`base_out`), separately for each reading code. I will also print the row-table columns, to find the true lag-1 dose deviation for checking the readout gain.

The check is that the edited shift on finger glucose is about a_ins times the dose-deviation readout, and close to zero for codes not tied to that drug. If so, a_ins is "mg/dL per u/h of deviation" and the gating works.

```python
Xs_, R_ = datasets[0]
print('R_ cols',R_.columns.tolist())
print('RA_ cols',RA_.columns.tolist(), RA_.shape)
o1=head_run(Xs_, R_, E1)
print(type(o1).__name__, [ (type(x).__name__, tuple(x.shape) if hasattr(x,'shape') else None) for x in (o1 if isinstance(o1,(tuple,list)) else [o1])])
pm_b,nll_b,P_b,hid_b=base_out[0]
pm1_=npv(o1[0]); nll1_=npv(o1[1])
cd=R_['code'].values
for c in [6,10,7,5]:
    m=cd==c
    print('code',c,'n',m.sum(),'mean shift %.3f sd %.3f'%((pm1_-pm_b)[m].mean(),(pm1_-pm_b)[m].std()),'dNLL %.4f'%((nll1_-nll_b)[m].mean()))
```