The resampling shows how much the risk depends on the size of the evaluation set.

| Stays in evaluation | P(cost > 0.005) at −2.6 | at −2.5 |
|---|---|---|
| 100 | 10% | 4.7% |
| 200 | 4% | 1.3% |
| 400 | 0.65% | 0.08% |
| 1000 | ≈0 | ≈0 |

The pooled mean cost is about 0.0031 at −2.6 and 0.0028 at −2.5. So −2.6 buys about 0.4 closer on insulin x2 and 0.2 on x1/2, at a small but real extra risk if the evaluation set is small.

A better trade-off may exist if the cost comes from a few glucose_finger rows where the edit moves the prediction a lot, e.g. readout noise outliers. Then capping the edit's push would cut the cost while barely changing the rollout means. If the cost is spread evenly, capping buys nothing.

To decide, I will tabulate the glucose_finger cost by the size of the edit's shift in predicted mean across all four tuning batches.

```python
parts=[]
for k in [13,14,15,17]:
    o_,e_=stay_cost(k); dpm=e_.predicted_mean.values-o_.predicted_mean.values; dn=e_.nll.values-o_.nll.values
    parts.append(pd.DataFrame({'fwd':k,'ev':o_.event.values,'dpm':dpm,'dn':dn,'obs':o_.observed.values,'pmo':o_.predicted_mean.values}))
PR_=pd.concat(parts,ignore_index=True); g_=PR_[PR_.ev=='glucose_finger'].copy()
tot=PR_.dn.sum(); print('total cost share by event',PR_.groupby('ev').dn.sum().div(tot).round(3).to_dict())
g_['ab']=pd.cut(g_.dpm.abs(),[-1e-9,.5,2,4,6,8,10,14,20,100])
t_=g_.groupby('ab',observed=True).agg(n=('dn','size'),cost=('dn','sum'),mean_dn=('dn','mean'),mdpm=('dpm','mean'))
t_['share']=t_.cost/tot; print(t_.round(4).to_string())
g_['sgn']=np.sign(g_.dpm); g_['resid_o']=g_.obs-g_.pmo
print('gf rows |dpm|>.5: corr(dpm,resid_o)',np.corrcoef(g_.dpm[g_.dpm.abs()>.5],g_.resid_o[g_.dpm.abs()>.5])[0,1].round(3),
      'OLS slope resid on dpm',(np.polyfit(g_.dpm[g_.dpm.abs()>.5],g_.resid_o[g_.dpm.abs()>.5],1)[0]).round(3))
```