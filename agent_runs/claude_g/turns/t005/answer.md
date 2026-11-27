Readings run from hour −1 to hour 24 on an hourly grid. Glucose_finger coverage and insulin use both rise over the stay, while MAP readings and norepinephrine stay flat.

Next I will study who gets what, i.e. the dosing policy:
- the share of insulin given in the same hour as a glucose_finger reading;
- the insulin rate against the glucose reading at h (and the lab value);
- the norepinephrine rate against MAP at h;
- how the presence of map_arterial relates to norepinephrine.

The question is whether the rate is a deterministic sliding scale of the last observed reading (as a real protocol would be), with or without noise, or depends on things that are not recorded. That decides whether adjusting for history is enough.

```python
Q = P[(P.index.get_level_values('h')>=0)].copy()
Q['ins'] = Q.insulin.fillna(0); Q['ne'] = Q.norepinephrine.fillna(0)
# insulin vs presence of glucose reading at same hour
print('P(insulin | gf at h):', Q[Q.glucose_finger.notna()].insulin.notna().mean(), ' P(insulin | no gf at h):', Q[Q.glucose_finger.isna()].insulin.notna().mean())
print('P(ne | map_art at h):', Q[Q.map_arterial.notna()].norepinephrine.notna().mean(), ' P(ne | no map_art):', Q[Q.map_arterial.isna()].norepinephrine.notna().mean())
# per-stay types
S = P.groupby(level='stay_id').count()
S['has_art'] = S.map_arterial>0; S['has_ne']=S.norepinephrine>0; S['has_ins']=S.insulin>0; S['has_gf']=S.glucose_finger>0
print(pd.crosstab(S.has_art, S.has_ne)); print(pd.crosstab(S.has_gf, S.has_ins))
print(S[['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']].describe().round(1).to_string())
# policy: insulin rate by glucose finger bins at same hour
g = Q[Q.glucose_finger.notna()].copy()
g['gbin'] = pd.cut(g.glucose_finger, [0,80,100,120,140,160,180,200,220,250,300,400,1000])
print(g.groupby('gbin', observed=True).agg(n=('ins','size'), p_ins=('insulin', lambda x: x.notna().mean()), mean_rate=('insulin','mean'), sd_rate=('insulin','std')).round(3).to_string())
```