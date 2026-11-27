Run 4 shows the encoding details:
- Weight is 80 kg for everyone, so norepinephrine dose = rate × 4.8 mg and carries no extra information.
- z = (x − median)/IQR, clipped to ±10. Fine bins come from per-code edges.
- The predicted mean is Σ p(bin) × centre, so glucose_finger predictions are bounded to 73–272.
- Norepinephrine rate edges top out at 1.0, the cap. Doubling pushes z far beyond the training range and possibly into bins never seen for that code.
- Halving insulin creates rates such as 0.25, 0.75 and 1.25 that never occur in training.

Next I'll build an hourly panel per stay with readings at hour h and drug rates at h+0.083. That lets me study three things: the dosing policy (how rate relates to the latest readings), the missingness of readings, and the finger/lab relationship. Together these show where the identifying variation for the causal effect could come from.

```python
d['h'] = np.floor(d.hour+1e-9).astype(int)
piv = d[d.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])].pivot_table(index=['stay_id','h'], columns='event', values='value', aggfunc='first')
drg = d[d.event.isin(['insulin','norepinephrine'])].pivot_table(index=['stay_id','h'], columns='event', values='rate', aggfunc='first')
P = piv.join(drg, how='outer').reset_index()
print(P.h.value_counts().sort_index().to_dict())
print(P.groupby('h')[['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']].count().T.to_string())
# duplicates per stay-hour?
print('dup events per stay-hour:', d.groupby(['stay_id','h','event']).size().max())
# paired finger vs lab same hour
pp = P.dropna(subset=['glucose_finger','glucose_lab'])
print('finger-lab same hour: mean diff', (pp.glucose_finger-pp.glucose_lab).mean(), 'sd', (pp.glucose_finger-pp.glucose_lab).std(), 'corr', pp[['glucose_finger','glucose_lab']].corr().iloc[0,1])
pa = P.dropna(subset=['map_arterial','map_cuff'])
print('art-cuff same hour: mean diff', (pa.map_arterial-pa.map_cuff).mean(), 'sd', (pa.map_arterial-pa.map_cuff).std(), 'corr', pa[['map_arterial','map_cuff']].corr().iloc[0,1])
```