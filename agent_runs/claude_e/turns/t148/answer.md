Against my truth estimates on these rows, the edit is partly right:

| Readout | Truth | Edited | Original |
|---|---|---|---|
| Insulin ×2 | −13.4 to −14.3 | −9.85 | −1.87 |
| Insulin ×½ | +6.7 to +7.2 | +4.13 | +0.76 |
| Norepinephrine ×2 | +4.9 | −0.80 | −0.51 |
| Norepinephrine ×½ | −2.45 | −1.91 | −0.70 |

**Insulin.** By hour, the edit is right at 1 hour (−3.35 against −3.04), and by 10–12 hours it nearly catches up. In between it builds too slowly, sitting at about 60% of the truth at 2–6 hours. That is the missing lag-2 and lag-3 direct effect. The mean over the window is what is scored, so scaling a_ins up by about 1.4 should match it. I will still check how much that costs in factual NLL.

**Norepinephrine.** Halving is about 70% of the way there, like insulin. Doubling has the wrong sign. My hypothesis is that doubled rates above 1.0, never seen in training, put the state where the linear readout extrapolates wrongly.

The test is to split the per-stay rollout shifts by the stay's recorded rate. If stays whose doubled rate stays ≤ 1 get correct positive shifts, and high-rate stays get the negative ones, the problem is out-of-range input. If low-rate stays are wrong too, the problem is elsewhere. I do the same split for insulin dose level.

```python
def per_stay(ed,vn,key):
    r=T10[(ed,vn)].merge(do[['stay_id','hour','edit','shift']].rename(columns={'shift':'orig'}),on=['stay_id','hour','edit'])
    G_=Gtu[key]; mr=[]
    for s in r.stay_id.unique():
        row=G_.loc[s]; v=row[[c for c in row.index if 12<=c<=23]].values.astype(float); mr.append((s,np.nanmean(np.where(v>0,v,np.nan)),np.nanmax(v)))
    M_=pd.DataFrame(mr,columns=['stay_id','mean_rate','max_rate'])
    ps=r.groupby('stay_id').agg(truth=('truth','mean'),edited=('shift','mean'),orig=('orig','mean'),n=('truth','size')).reset_index().merge(M_,on='stay_id')
    return ps
ps=per_stay('ne_x2','dN_use','ne')
ps['band']=pd.cut(ps.max_rate,[0,0.5,0.75,1.01])
print('NE x2 by stay max rate (h12-23):'); print(ps.groupby('band').agg(stays=('n','size'),rows=('n','sum'),truth=('truth','mean'),edited=('edited','mean'),orig=('orig','mean')).round(2))
ph=per_stay('ne_half','dN_use','ne'); ph['band']=pd.cut(ph.max_rate,[0,0.5,0.75,1.01])
print('NE half by stay max rate:'); print(ph.groupby('band').agg(stays=('n','size'),truth=('truth','mean'),edited=('edited','mean'),orig=('orig','mean')).round(2))
pi_=per_stay('insulin_x2','d_avg','ins'); pi_['band']=pd.cut(pi_.mean_rate,[0,1.5,3,5,20])
print('insulin x2 by stay mean dose:'); print(pi_.groupby('band').agg(stays=('n','size'),truth=('truth','mean'),edited=('edited','mean'),orig=('orig','mean')).round(2))
print('corr(edited,truth) per stay: ins %.2f ne %.2f'%(np.corrcoef(pi_.edited,pi_.truth)[0,1],np.corrcoef(ps.edited,ps.truth)[0,1]))
```