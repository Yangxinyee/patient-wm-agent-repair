The threshold comparison confirms the null. Glucose 1–6 hours after a reading just above 150 (insulin started, about 1.05 U/h) differs from just below 150 (no insulin) by −0.01 to −0.81 mg/dL. The SEs are 0.3–0.6, so none of these is clearly non-zero.

Together with the rate experiment, the records say insulin does essentially nothing to glucose in this simulated ICU, whether started or raised. My target for both insulin readouts is therefore about 0.

Next is norepinephrine. In the example stay its rate changes smoothly from hour to hour, which suggests titration from the previous rate rather than a fixed scale. I will look at:
- which MAP reading drives starting it (arterial vs cuff, and the threshold);
- how the hour-to-hour rate change depends on MAP and on the previous rate.

The aim is to find the random component of the dose, as I did for insulin.

```python
ne = xx.copy()
ne['Mtrig'] = ne.map_arterial.where(ne.map_arterial.notna(), ne.map_cuff)
ne['ne_prev'] = ne.groupby('stay_id').norepinephrine.shift(1)
# P(NE | MAP bins), split by whether NE was running previous hour
for prevon in [False, True]:
    d = ne[(ne.h>=0) & (ne.ne_prev.notna()==prevon)]
    print("prev NE on:", prevon)
    print(d.groupby(pd.cut(d.Mtrig, [0,50,55,60,62,64,65,66,68,70,75,80,90,200])).norepinephrine.apply(lambda s: s.notna().mean()).round(3).to_dict())
# P(NE) when no MAP measured at all
d = ne[(ne.h>=0) & ne.map_arterial.isna() & ne.map_cuff.isna()]
print("no MAP at hour: P(NE)", d.norepinephrine.notna().mean(), len(d))
# first NE: MAP at that hour
first = ne[(ne.h>=0) & ne.norepinephrine.notna() & ne.ne_prev.isna()]
print(first[['Mtrig','map_arterial','map_cuff','norepinephrine']].describe().to_string())
```