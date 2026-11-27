Run 6 shows that dosing is irregular in two sample stays:
- Insulin appears only in some hours.
- Norepinephrine rates swing widely from hour to hour.
- Readings are missing in many hours.

Next I'll characterise the policy quantitatively:
- the probability of an insulin record at h+0.083, and its rate, as a function of the current glucose_finger and the previous rate;
- the same for norepinephrine against MAP.

If the dose is a tight function of recorded readings, conditioning on history gives little identifying variation. A lot of unexplained dose variation would be either exogenous noise (good for identification) or tracking of unrecorded state (confounding). I'll look at R² and the shape of the dependence.

```python
x = P[P.h>=0].copy()
x['ins_on'] = x.insulin.notna().astype(int)
x['ne_on'] = x.norepinephrine.notna().astype(int)
x['gf_bin'] = pd.cut(x.glucose_finger, [0,100,120,140,150,160,170,180,190,200,220,250,400])
print(x.groupby('gf_bin').agg(n=('ins_on','size'), p_ins=('ins_on','mean'), rate_mean=('insulin','mean'), rate_sd=('insulin','std')).to_string())
x['gf_missing'] = x.glucose_finger.isna()
print(x.groupby('gf_missing').agg(n=('ins_on','size'), p_ins=('ins_on','mean'), rate_mean=('insulin','mean')).to_string())
x['ins_prev_on'] = x.insulin_prev.notna()
print(x.groupby(['ins_prev_on','gf_missing']).agg(n=('ins_on','size'), p_ins=('ins_on','mean'), rate_mean=('insulin','mean')).to_string())
x['map_bin'] = pd.cut(x.map_last, [0,50,55,60,65,70,75,80,85,90,200])
print(x.groupby('map_bin').agg(n=('ne_on','size'), p_ne=('ne_on','mean'), rate_mean=('norepinephrine','mean'), rate_sd=('norepinephrine','std')).to_string())
```