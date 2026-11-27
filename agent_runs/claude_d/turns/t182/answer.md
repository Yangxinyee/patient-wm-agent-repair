The NE deviation from its MAP-history expectation did not help much. The MAP history explains 77% of the NE rate variance, leaving a deviation with sd 0.14, about the same as the one-step surprise. With_code decodes it with R² of only 0.16 (full state 0.22).

The joint residual regression on those deviations at lags 1 to 3 gives 3.1, 0.2, -2.2 (arterial), with cuff similar. So the model's factual deficit is mostly a missing response to the most recent rate change. Longer-run effects are carried forward through the real readings that follow.

That suggests a working picture for both drugs:

- What is missing is the short-lag response.
- In a rollout, the model's own propagation through its sampled readings would supply the later lags, if the first lags were right.

To check this for insulin, I extend the residual regression to the insulin surprise at lags 1 to 6. If the deficits beyond lag 2 or 3 are near zero, fixing the first one or two lags at the head should be enough. Otherwise the head edit alone will under-shoot the cumulative effect.

```python
E2 = PT_tr.set_index(['stay_id','t'])['eps_ins']
for k in range(1,7):
    ix = pd.MultiIndex.from_arrays([RD2['stay_id'].values, (RD2['hour']-k).values])
    RD2[f'ei{k}'] = E2.reindex(ix).fillna(0).values
cols6 = [f'ei{k}' for k in range(1,7)]
for code in ['glucose_finger','glucose_lab']:
    print(code, reg(ev==code, cols6))
print('both', reg(isg, cols6))
print('var eps1 gluc rows', round(float(RD2.loc[isg,'ei1'].var()),3), 'var Dne1 map rows', round(float(RD2.loc[ism,'D_ne_1'].var()),4))
```