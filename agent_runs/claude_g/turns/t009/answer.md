The insulin multiplier looks like pure per-hour randomisation:
- lag-1 autocorrelation of log m is 0.002;
- the between-stay variance of mean log m (0.0058) is about what sampling noise alone would give (0.107 / ~16 ≈ 0.0067).

So given the glucose reading, the extra dose variation is a natural experiment, and comparing hours with the same glucose but different m identifies insulin's causal effect.

Before relying on this I want to:
1. Confirm that m is unrelated to other covariates (MAP, norepinephrine, glucose_lab, hour, previous glucose and its trend).
2. See how the rare insulin doses given without a glucose_finger reading at h were chosen.
3. Look at the norepinephrine policy the same way: rate against MAP (arterial or cuff), the cap at 1.0, and a multiplier.

If m correlates with nothing, I will treat it as randomised.

```python
# 1. m vs covariates
Q['gf_prev'] = Q.groupby(level='stay_id').glucose_finger.shift(1)
Q['map_a'] = Q.map_arterial; 
cols = ['gf_prev','glucose_lab','map_arterial','map_cuff','ne']
gg = gg.join(Q[['gf_prev']], how='left', rsuffix='_q') if 'gf_prev' not in gg else gg
gg['hh'] = gg.index.get_level_values('h')
for c in ['glucose_finger','gf_prev','glucose_lab','map_arterial','map_cuff','ne','hh','ins_prev']:
    s = gg[[c,'lm']].dropna()
    print(c, len(s), round(np.corrcoef(s[c], s.lm)[0,1],4))
# 2. insulin without gf at h
nog = Q[Q.insulin.notna() & Q.glucose_finger.isna()]
print('n insulin without gf:', len(nog)); print(nog.head(20).to_string())
```