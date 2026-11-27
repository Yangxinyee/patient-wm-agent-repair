Head 3 skips the u token when the bundle holds a single insulin atom (316 bundles); then head 1 attends u instead. Bundles where head 3 attends u hold about 1.85 insulin atoms each. The drug/non-drug gate from head 3's output is clean: drug minimum 0.77 vs non-drug maximum −0.80. But the insulin rate decoder still has R² of only 0.23, even on clean bundles.

I suspect my per-bundle target is wrong rather than the signal missing. Two-atom insulin bundles probably pair the end of the previous infusion with the start of a new one, and `bundle_rates` keeps whichever atom comes last. Next I'll inspect the insulin atoms' kind, rate and dose fields in one- and two-atom bundles. If kinds like start/stop with different rates appear, I'll redefine the target as the rate of the infusion actually running and refit.

```python
ai=at[at.event=='insulin'].copy(); ai['bi']=[ridx[x] for x in ai['row'].values]
cnt=ai.groupby('bi').size(); ai['n']=ai['bi'].map(cnt)
print(ai['kind'].value_counts().to_dict())
print(ai.groupby(['n','kind'])[['rate','dose']].agg(['mean','count']).round(3).to_string()[:900])
ex=ai[ai.n==2].head(6); print(ex[['stay_id','hour','minute','kind','rate','dose','value']].to_string())
```