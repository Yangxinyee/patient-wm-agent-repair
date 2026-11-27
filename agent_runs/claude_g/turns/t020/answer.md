Insulin's dose effect is about 0 at every lag from 1 to 12 hours, both overall and for dose hours ≥ 12. The coefficients sit at +0.0 to +0.16 mg/dL per u/h with SE 0.04–0.12. A few late-window values (lags 7–11, h ≥ 12) reach +0.27 to +0.37 with large SEs, and none is negative. If anything, there is a hint of a tiny positive bias, possibly from residual policy structure such as dependence on the previous reading. Even taken at face value, it would give an insulin x2 readout of well under +1 mg/dL.

Current belief:
- Insulin: inert. The true x2 and x1/2 readouts are about 0, possibly within ±0.5 mg/dL.
- Norepinephrine: inert. The true readouts are about 0, at most −0.2 mmHg.

Next I want to see the original model's answers. I will save notes first, then pick tuning stays that actually receive insulin and norepinephrine at or after hour 12 (only those enter the readouts), and list them for a dose_edit request. Then I can see how far the model is from 0 and which stay-hours the readout counts.

```python
P_tu = make_panel(tu)
Qt = P_tu[P_tu.index.get_level_values('h')>=0].copy()
Qt['ins'] = Qt['insulin'].fillna(0); Qt['nev'] = Qt['norepinephrine'].fillna(0)
late = Qt[Qt.index.get_level_values('h')>=12]
st_ins = late.groupby(level='stay_id')['ins'].apply(lambda x: (x>0).sum())
st_ne = late.groupby(level='stay_id')['nev'].apply(lambda x: (x>0).sum())
st_art = late.groupby(level='stay_id')['map_arterial'].count()
info = pd.DataFrame({'ins12':st_ins, 'ne12':st_ne, 'art12':st_art})
print(len(info), (info.ins12>0).sum(), (info.ne12>0).sum(), ((info.ne12>0)&(info.art12>0)).sum(), ((info.ins12>0)&(info.ne12>0)&(info.art12>0)).sum())
both = info[(info.ins12>0)&(info.ne12>0)&(info.art12>0)].index.tolist()
ins_only = info[(info.ins12>0)&~((info.ne12>0)&(info.art12>0))].index.tolist()
sel100 = both[:60] + ins_only[:40]
print(len(sel100)); print(json.dumps([int(s) for s in sel100]))
notes = """
## Data findings (session 1)
- All stays 25.17 h; readings on integer hours; infusions start h+0.0833 and last 1 h.
- Insulin policy: base by latest glucose reading (finger, else lab): >150:1, >180:2, >220:3, >260:4, >300:6 u/h, times
  iid multiplier m (lognormal-ish, sd~1/3), rounded to 0.5. m uncorrelated with everything (lag1 corr 0.002, no stay effect).
- NE policy: start iff MAP<65 (sharp), titrate ratio depends on MAP (<65 up, 65-75 flat, >75 down ~0.8), cap 1.0.
- CAUSAL: insulin dose-margin effect on glucose_finger ~0 at lags 1..12 (coef ~ +0.05+-0.05 mg/dL per u/h).
  RD at 150 threshold (start insulin): no jump. => INSULIN INERT.
- NE: DML margin effect on map_arterial -0.33+-0.08 mmHg per mcg/kg/min at lag1, ~0 later; RD at MAP 65 start: no jump.
  => NE INERT (at most -0.2 mmHg for x2).
"""
open('/work/notes.md','a').write(notes)
```