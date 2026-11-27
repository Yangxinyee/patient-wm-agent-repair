That run had a bug. Where a finger and a lab reading fell in the same hour, I set the previous-reading time to missing, so those rows got the full level shift instead of the increment. That is why "full" rose to 0.104.

I'll recompute with the previous reading time defined as the last glucose reading strictly before the current hour.
- If the band-free constant-reference version then costs around 0.003 or less, I can avoid band compensation.
- If it still costs about 0.01, the edit must compare the dose with the state's band.

```python
gg2 = gg.sort_values(['stay_id','hour']).copy()
hrs_by = gg2.groupby('stay_id').ih.apply(lambda s: np.unique(s.values)).to_dict()
def prev_h(s_,t_):
    a_=hrs_by[s_]; k_=np.searchsorted(a_,t_)-1
    return a_[k_] if k_>=0 else np.nan
gg2['tlast']=[prev_h(s_,t_) for s_,t_ in zip(gg2.stay_id,gg2.ih)]
print('full', round(cost(R,gg2),4), 'dev', round(cost(RD,gg2),4))
for dref in [1.0,1.5,2.0,2.5]:
    print('dref',dref, round(cost(np.where(R>0,R-dref,0.0),gg2),4))
```