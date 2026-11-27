Insulin is recorded only at hours that have a glucose reading (finger or lab, 100%). The first insulin in a stay always comes at a finger glucose of at least 150 mg/dL, which points to a threshold rule. Rates come in 0.5 u/h steps with a long right tail, perhaps a log-normal multiplier rounded to 0.5.

Next I'll characterise the rate given the glucose reading at the same hour (finger if present, otherwise lab):
- the conditional mean and spread of the rate by glucose bin;
- the probability of continuing insulin, once started, at each glucose level (the stopping rule);
- whether the previous rate matters (stickiness).

I'll do the same for norepinephrine and MAP. Knowing the exact policy shows how much exogenous dosing noise there is to identify effects from.

```python
P['g_now'] = P.glucose_finger.fillna(P.glucose_lab)
P['started_ins'] = P.groupby('stay_id').insulin.transform(lambda s: s.notna().cumsum().shift(1).fillna(0)>0)
sub = P[P.g_now.notna()]
sub = sub.assign(gbin=pd.cut(sub.g_now,[0,100,120,140,150,160,180,200,220,250,400]))
t = sub.groupby(['started_ins','gbin']).agg(n=('h','size'), p_ins=('insulin', lambda s: s.notna().mean()),
                                           mean_rate=('insulin','mean'), sd_rate=('insulin','std'),
                                           mean_lograte=('insulin', lambda s: np.log(s).mean()), sd_lograte=('insulin', lambda s: np.log(s).std()))
print(t.to_string())
```