Batch 0 of the train states arrived: 100 stays, 5362 bundles. Before spending five more forward passes, I'll check the pipeline on this batch:
- build the reading rows and compute the model's predicted means with the local head;
- estimate the model's one-step coefficients on dose deviation;
- check how well the current insulin deviation can be read linearly from the head's query vector, using cross-validation by stay.

Two outcomes would justify requesting the remaining batches: a cross-validated R² for lag-1 deviation of at least 0.4, and model coefficients broadly consistent with the 200-stay tuning estimates.

```python
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
b0 = batches[0]
Xs5, R5 = build_reading_rows('/work/fwd_5/states_original.npz', Gtr)
pm5, nll5, P5 = head_eval(torch.as_tensor(np.asarray(Xs5)) if not torch.is_tensor(Xs5) else Xs5, R5, nz=16)
R5['pm'] = pm5; R5['nll'] = nll5
print(R5.event.value_counts().to_dict(), 'local NLL', nll5.mean())
Db0 = build_ins_design2(Gtr, b0); Db0['sid'] = np.array(b0)[Db0.s.values.astype(int)]
g5 = R5[R5.event=='glucose_finger'][['stay','hour','pm','prev','dt']].rename(columns={'stay':'sid'})
g5['hour'] = g5.hour.round().astype(int); Db0['hour'] = Db0.hour.round().astype(int)
Db0 = Db0.merge(g5, on=['sid','hour'], how='left'); Db0 = Db0[Db0.pm.notna()].reset_index(drop=True)
print('design rows with pm', len(Db0))
b_y, s_y, _ = my_reg(Db0, 'y'); b_m, s_m, _ = my_reg(Db0, 'pm')
dv = [f'dev{k}' for k in range(1,7)]
print('truth(b0) dev:', b_y[dv].round(2).values, ' se', s_y[dv].round(2).values)
print('model(b0) dev:', b_m[dv].round(2).values, ' se', s_m[dv].round(2).values)
# decodability of dev lags from q (mean over z draws) at these glucose rows
Qg = q_from_rows(Xs5 if torch.is_tensor(Xs5) else torch.as_tensor(Xs5), torch.as_tensor(Db0.prev.values.astype(int)), torch.as_tensor(Db0.dt.values, dtype=torch.float64), nz=8, seed=1).mean(0).numpy()
grp = Db0.sid.values
for k in range(1,5):
    y = Db0[f'dev{k}'].values.astype(float)
    pred = cross_val_predict(RidgeCV(alphas=np.logspace(-2,3,12)), Qg, y, cv=GroupKFold(5), groups=grp)
    print(f'dev{k}: sd {y.std():.3f}  CV R2 from q: {1-np.mean((y-pred)**2)/y.var():.3f}')
```