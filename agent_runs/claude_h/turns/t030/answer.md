Insulin is null in every subgroup: base level 1 to 6, with or without norepinephrine, with or without an arterial line, and early or late hours. The largest estimate is +0.18 ± 0.10 at base 6, and the direction is not consistent. Norepinephrine's weak negative long-horizon association is unstable: −0.36 ± 0.16 in hours before 12 but +0.05 ± 0.11 in hours from 12 on, and noisy across MAP levels. That looks like chance.

Measured against realistic effect sizes, both drugs act as "absent". I can bound the immediate norepinephrine effect at under about 0.3 mmHg per mcg/kg/min, and insulin's at under 0.1 mg/dL per u/h at any horizon. So the target readouts are about 0, and the original model is already close (+0.07, −0.03, −0.03, +0.01).

A natural improvement is to make the model exactly insensitive to dose values, so that the rollouts under the edits are identical and the shift is exactly 0. That only works if removing dose information does not hurt factual fit, which holds only if doses carry no information about future readings beyond the readings already recorded.

One place doses could carry extra information: norepinephrine is titrated even in hours with no MAP reading. If the clinician then used the unobserved MAP, the rate change would predict the next reading. Next I will test whether, in hours without a MAP reading, the norepinephrine rate change predicts the next MAP, and similarly whether the insulin rate predicts the next glucose beyond the readings in any subset. I am also updating the notebook.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 14-28: norepinephrine; model baseline
- NE policy: titration. Start if MAP<~64.5 (rate higher if MAP lower); continuing: rate_t = g(MAP_t, rate_{t-1}) + iid innovation (sd .138, autocorr -.001,
  uncorrelated w/ MAP trend, cuff reading, prior MAPs). Shock persists in later rates R=.67,.48,.35,...
- NE LP of MAP_{t+k}-MAP_t on innovation: k=1 +0.14(.09), later -0.05..-0.6 (se .1-.25). Summary avg6-12 -0.28(.14), cuff -0.29(.13); hours<12 -0.36(.16), >=12 +0.05(.11). => weak/inconsistent; treat as ~0.
- Insulin: LP k=1..12 all ~0 (se .04-.11), RD at 150/180/220/260 ~0, by subgroup ~0. => insulin effect ABSENT.
- FWD 1 (dose_edit original, tuning sel1 100 stays, /work/sel1.json): ins x2 +0.066, x1/2 -0.032, ne x2 -0.027, ne x1/2 +0.009 (all ~0; stay-clustered se ~.04/.02).
  x2 and x1/2 shifts correlate +0.58 per stay-hour => partly perturbation noise.
- Current belief: true effects ~0 for both drugs; original model already ~0. Option: make model exactly dose-invariant (shift=0) if NLL holds.
""")
# NE: no MAP reading at t, on norepi at t-1 and t
dd = P[(P['nor_prev']>0) & P['ma'].isna() & P['mc'].isna()].copy().reset_index()
dd['dr'] = dd['nor'] - dd['nor_prev']
dd['ma_f1'] = dd['ma_f1']; dd['mc_f1']=dd['mc_f1']
dd['mapf1'] = dd['ma_f1'].combine_first(dd['mc_f1'])
mp = P['map_any'].groupby(level=0).shift(1).reset_index().rename(columns={'map_any':'mapp'})
dd = dd.merge(mp, on=['stay_id','ih'], how='left')
dd['pbin'] = (dd['nor_prev']*20).round().astype(int)
sub = dd[dd.mapp.notna() & dd.mapf1.notna()].copy()
sub['mpb'] = (sub.mapp.clip(20,110)//2).astype(int); sub['cell'] = sub.mpb*100+sub.pbin
sub['y'] = sub.mapf1 - sub.mapp
b,se,n = fe_ols(sub, 'y', ['dr'], 'cell'); print("no-MAP hours: next MAP change on rate change:", b[0].round(3), se[0].round(3), n)
print("no-MAP hours rate change sd:", dd.dr.std().round(3), " vs with MAP:", (P['nor']-P['nor_prev'])[(P['nor_prev']>0)&P['map_any'].notna()].std().round(3))
```