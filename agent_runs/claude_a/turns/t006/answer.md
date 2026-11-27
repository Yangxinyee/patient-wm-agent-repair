Run 5 points to selection in who gets measured:
- Finger and lab glucose agree when taken in the same hour (mean difference −0.02, SD 9). So finger readings averaging 18 mg/dL higher is selection: finger sticks happen more when glucose is high or on insulin.
- Arterial MAP reads about 3 mmHg below cuff MAP (SD 6.7).
- The number of finger readings and insulin records rises over the stay, while norepinephrine use peaks around hour 5 and then declines.

Next I'll examine the dosing policy directly. I'll regress the insulin rate at h+0.083 on the latest glucose readings (finger and lab at hour h) and the previous rate, and do the same for norepinephrine on MAP. I also want to see how much residual variation exists. Exogenous variation in dose, given the recorded history, is what would let me identify causal effects. A near-deterministic sliding scale would make identification hard; a dose that depends on things not in the record would mean confounding.

```python
P = P.sort_values(['stay_id','h']).reset_index(drop=True)
g = P.groupby('stay_id')
for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']:
    P[c+'_prev'] = g[c].shift(1)
# last observed glucose (finger or lab) at or before hour h
P['glu_now'] = P.glucose_finger.fillna(P.glucose_lab)
P['glu_last'] = g['glu_now'].ffill() if False else P.groupby('stay_id')['glu_now'].ffill()
P['map_now'] = P.map_arterial.fillna(P.map_cuff)
P['map_last'] = P.groupby('stay_id')['map_now'].ffill()
# show a stay with both drugs
sid = P[(P.insulin.notna())].stay_id.unique()
sid2 = [s for s in sid[:400] if P[(P.stay_id==s)].norepinephrine.notna().sum()>5][:2]
for s in sid2:
    print(P[P.stay_id==s][['h','glucose_finger','glucose_lab','insulin','map_arterial','map_cuff','norepinephrine']].to_string())
```