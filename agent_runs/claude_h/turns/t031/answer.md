In hours without a MAP reading, the norepinephrine rate change does not predict the next MAP (−0.22 ± 0.55 per mcg/kg/min, from about 11k hours). Its spread is the same as in hours with a reading (0.145 vs 0.149). So I see no sign that doses carry hidden information about the patient beyond what is recorded. That suggests removing the model's dose sensitivity would not cost factual fit, which I will verify with the tool.

Before deciding between "leave as is" and "make exactly dose-invariant", I want one more independent check on norepinephrine, using different variation from the titration noise: a fuzzy regression discontinuity at the start threshold. Patients just below the MAP cut get started on norepinephrine and then keep it for hours; those just above mostly do not. A real effect of either sign should open a MAP gap between the groups over the following 12 hours. First I will locate the exact threshold, then estimate the jump in later MAP changes, scaled by the jump in cumulative norepinephrine dose.

```python
s0 = P[(P['nor_prev']==0) & P['ma'].notna()].copy().reset_index()
s0['start'] = (s0['nor']>0).astype(float)
print(s0.groupby(s0['ma'].round())['start'].agg(['mean','size']).loc[58:72].round(3).T.to_string())
# cumulative dose over next k hours
for k in range(1,13):
    s0[f'cum{k}'] = sum(s0.merge(P[['nor']].groupby(level=0).shift(-j).fillna(0).reset_index().rename(columns={'nor':'nf'}), on=['stay_id','ih'], how='left')['nf'].values for j in range(0,k))
s0 = s0.merge(P[[f'ma_f{k}' for k in range(1,13)]].reset_index(), on=['stay_id','ih'], how='left', suffixes=('_x',''))
for k in [1,2,3,6,9,12]:
    s0[f'dM{k}'] = s0[f'ma_f{k}'] - s0['ma']
```