Every stay is exactly 25.17 hours long. Measurements fall exactly on the hour and drug records exactly 5 minutes past it, so the record is a clean hourly grid.

About 61% of stays get insulin and 43% get norepinephrine, apparently independently. Glucose_finger counts per stay are bimodal (around 8 or around 22–26), and map_arterial appears in about 48% of stays. That suggests:
- stays on insulin get hourly finger-sticks while others are checked less often;
- stays on norepinephrine get an arterial line.

Next I'll check those links and print one norepinephrine stay in full. Then I'll study the dosing policy: how the insulin rate at hour t + 5 min relates to the glucose read at hour t and to the previous rate. If the rate is a deterministic or near-deterministic function of observed readings plus independent noise, the effect is identifiable by adjusting for the history. Any random component (noise, rounding, clinician style) is then the usable variation.

```python
print(pd.crosstab(cnt.insulin>0, pd.cut(cnt.glucose_finger,[-1,5,10,15,20,30])))
print(pd.crosstab(cnt.norepinephrine>0, pd.cut(cnt.map_arterial,[-1,0,10,20,30])))
print(pd.crosstab(cnt.norepinephrine>0, pd.cut(cnt.map_cuff,[-1,5,10,15,20,30])))
s = tr[tr.stay_id.isin(cnt.index[(cnt.norepinephrine>0)&(cnt.insulin>0)][:1])]
print(s.drop(columns=['rate_unit','dose_unit']).to_string())
```