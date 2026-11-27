The threshold is sharp and acts on the unrounded finger reading: 0% get insulin at 149, 48% at exactly 150 (half of the values that round to 150), 100% at 151. When there is no finger stick in the hour, insulin follows a lab glucose ≥ 150 instead. So whether insulin starts at all is a deterministic function of the recorded reading. That is good for identification: the choice depends only on what is in the record.

Next I will model the rate given the triggering reading (finger, else lab). I will regress log rate on the glucose excess over 150 and on history (previous glucose, previous rate, hour). Then I will check whether the residual persists within a stay. A persistent per-stay scale would suggest either a clinician habit or a patient trait that might also affect glucose. Independent hourly noise would be a clean source of dose variation.

```python
xx = x[(x.h>=0)].copy()
xx['Gtrig'] = xx.glucose_finger.where(xx.glucose_finger.notna(), xx.glucose_lab)
mi = xx[xx.insulin.notna()].copy()
mi['lr'] = np.log(mi.insulin)
mi['ex'] = mi.Gtrig - 150
# binned mean of rate vs excess
mi['eb'] = pd.cut(mi.ex, [-1,5,10,20,30,40,60,80,100,130,160,200,400])
print(mi.groupby('eb').insulin.agg(['mean','median','std','count']).to_string())
print(mi.groupby('eb').lr.agg(['mean','std']).to_string())
```