Sampling z does not reduce the mismatch: finger glucose is still off by about 5.7 mg/dL. So the remaining error is structural, not noise from the 4 z draws.

Type 1 now makes sense. With types indexed as atom kind × 5 + event kind, a measurement WORLD atom is type 1.

The suspects are therefore the register terms or the state and time alignment. Next I regress my per-reading error on candidate explanations:
- has_last;
- time since the last reading of the same code;
- the gap dt to the previous bundle;
- the last value and its bin;
- whether the previous bundle was a drug bundle;
- hour of stay;
- whether a finger and a lab glucose share the bundle.

A clear dependence on one of them will show which input I have wrong.

```python
pm_, _, _ = run2()
fa4['err'] = pm_ - tgt
fa4['prev_drug'] = (S['has_insulin'][fa4.prev.values] + S['has_norepinephrine'][fa4.prev.values]) > 0
fa4['n_in_bundle'] = fa4.groupby(['stay_id','hour']).event.transform('size')
gl_b = fa4[fa4.event=='glucose_lab'][['stay_id','hour']].assign(gl_same=1)
fa4 = fa4.drop(columns=[c_ for c_ in ['gl_same'] if c_ in fa4]).merge(gl_b, on=['stay_id','hour'], how='left'); fa4['gl_same']=fa4.gl_same.fillna(0)
for ev in ['glucose_finger','map_arterial']:
    d4 = fa4[fa4.event==ev]
    print(ev, 'corr err with:', {c_: round(np.corrcoef(d4['err'], d4[c_].astype(float))[0,1],3) for c_ in ['has','age_s','dt','zl','lb','prev_drug','hour','n_in_bundle','gl_same','observed','predicted_mean']})
    print(d4.groupby('prev_drug').err.agg(['mean','std','size']).round(2).to_dict(), d4.groupby(pd.cut(d4.age_s/3600,[-.1,0.5,1.5,2.5,30])).err.agg(['mean','std','size']).round(2).to_dict())
```