Two independent 4-draw estimates from my head differ by MAE 1.55. So the tool's sampling noise accounts for roughly 1.1 units of MAE, not the whole 3.5. The NLL gap is also systematic: about 1.645 for me against 1.549 for the tool.

So the tool has extra information, and the bundle-size pattern suggests it comes from other readings in the same bundle. I will test this directly on bundles that hold two related readings:
- finger with lab glucose;
- arterial with cuff MAP.

For each pair, I correlate the gap between the tool and my head on one reading with the surprise in the other, its observed value minus my prediction. A clear positive correlation in one direction, or both, would show within-bundle conditioning and its order. Near-zero correlations would rule this out.

```python
F['mine']=(p_draws(16,7)[:,1:]*cen).sum(1); F['d']=F.predicted_mean-F['mine']; F['r']=F.observed-F['mine']
pv=F.pivot_table(index=['stay_id','hour'],columns='event',values=['d','r'])
for a_,b_ in [('glucose_finger','glucose_lab'),('glucose_lab','glucose_finger'),('map_arterial','map_cuff'),('map_cuff','map_arterial'),('glucose_finger','map_cuff'),('map_cuff','glucose_finger')]:
    sub=pv[[('d',a_),('r',b_)]].dropna()
    if len(sub)>10:
        c=np.corrcoef(sub[('d',a_)],sub[('r',b_)])[0,1]; sl=np.polyfit(sub[('r',b_)],sub[('d',a_)],1)[0]
        print(f"d[{a_}] vs surprise[{b_}]: n={len(sub)} corr={c:.3f} slope={sl:.3f}")
```