The head-only decoder fails. The model's teacher-forced response to insulin is only −0.26 per U/h at lag 1 and about 0 after that, against the data's −1.99, −1.47, −0.89, …. For MAP it is +0.77 at lag 1 and 0 after. A linear decoder of the needed correction from the head input reaches only R² ≈ 0.3 with slope 0.28. So the state seems to forget a dose within one step.

That suggests a cheaper mechanism. Correct only the most recent dose's deviation, at lag 1 (maybe lag 2), and let the rollout propagate the shifted draws. To judge it I need two things:
1. Per-lag decodability of the dose change from the head input. If lag 1 decodes well (R² > 0.8) and lags ≥ 2 do not, the lag-1 route is feasible.
2. The policy function's signature, needed to build factual deviations.

```python
import inspect
for fn_ in ['pfun','dose_lag','pol_lag','hourly','true_shift_rows','ols_']:
    try: print(fn_, inspect.signature(globals()[fn_]))
    except Exception as ex: print(fn_, ex)
iv=np.where(GL&np.isin(DC['arm'],ins_arms))[0]
WCi=wc_of(DC,W,iv)
sv=np.array([s in test_s for s in DC['si'][iv]])
for k in range(4):
    y_=dIns[iv,k]
    for alpha in [1.0]:
        w_,b_=ridge(WCi[~sv],y_[~sv],alpha); pr=WCi[sv]@w_+b_
        print(f'lag {k+1}: test R2 {1-np.var(y_[sv]-pr)/np.var(y_[sv]):.3f}  slope {np.polyfit(y_[sv],pr,1)[0]:.3f}  sd target {y_[sv].std():.2f}')
# also absolute dose at lag1 (interventional + factual rows)
iall=np.where(GL&~np.isin(DC['arm'],ne_arms))[0]
y_=DinsA[iv,0]; w_,b_=ridge(WCi[~sv],y_[~sv],1.0); pr=WCi[sv]@w_+b_
print('abs dose lag1 R2',1-np.var(y_[sv]-pr)/np.var(y_[sv]))
res_f=(DC['value']-DC['pm'])[GL&(DC['arm']==0)]; print('factual glucose resid sd',np.nanstd(res_f), 'MAP resid sd',np.nanstd((DC['value']-DC['pm'])[MP&(DC['arm']==0)]))
```