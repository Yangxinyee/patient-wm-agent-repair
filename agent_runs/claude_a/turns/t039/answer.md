The 4-draw scatter explains only about 2 mg/dL of glucose error and 0.9 mmHg of MAP error. My heads still miss by 7 mg/dL and 2 to 4.8 mmHg, so something systematic is still wrong. The conflicting age-unit preferences (hours for glucose, seconds for MAP) suggest that the age feature is not the real problem but is soaking up some other error.

Next I will look at how the residual (mine minus the model's) is structured: its correlation with the observed and last values, my own prediction, the hour, first readings, and whether insulin or NE was in the previous bundle. I will also compare my bin NLL with the reported NLL, and try `last_status` row 1 for readings that have a last value. If the residual tracks the last value or my own prediction, the register or the scale of the transition term is wrong. If it tracks drug bundles, the context is wrong.

```python
p_, vbt = head_probs('gelu', dts_opts['s'], 'L2', 1.0, 'left', 32)
m_ = (p_[:, 1:] * centM[codes_]).sum(1)
res_ = (m_ - obs_pm).numpy()
Rk = R.copy(); Rk['mine'] = m_.numpy(); Rk['res'] = res_
Rk['has'] = has; Rk['zv'] = zv
prevrow = R.row_prev.values
Rk['prev_ins'] = S['has_insulin'][prevrow]; Rk['prev_ne'] = S['has_norepinephrine'][prevrow]; Rk['prev_nat'] = S['n_atoms'][prevrow]
# observed bin and nll
ob = np.zeros(len(R), np.int64)
for ev in R.event.unique():
    m = (R.event == ev).values
    ob[m] = fbin(ev, 'value', R.observed.values[m])
Rk['mynll'] = -torch.log(p_[torch.arange(len(R)), torch.tensor(ob)]).numpy()
for evn in ['glucose_finger', 'map_arterial']:
    q_ = Rk[Rk.event == evn]
    print(evn, 'n', len(q_), 'mean res', q_.res.mean().round(3), 'rmse', np.sqrt((q_.res**2).mean()).round(3), 'nll obs', q_.nll.mean().round(4), 'mine', q_.mynll.mean().round(4))
    for c in ['observed', 'lastv', 'mine', 'hour', 'has', 'prev_ins', 'prev_ne', 'prev_nat', 'dt']:
        x_ = q_[c].astype(float).values; ok = ~np.isnan(x_)
        print('   corr res~', c, np.corrcoef(x_[ok], q_.res.values[ok])[0, 1].round(3))
    print(q_.groupby('prev_nat').res.agg(['mean', 'std', 'count']).round(2).to_string())
    print(q_.groupby(q_.hour.clip(-2, 3)).res.agg(['mean', 'std', 'count']).round(2).to_string())
```