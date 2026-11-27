Run 3 shows several structural facts:
- Every stay lasts exactly 25.17 h.
- Readings are on the hour; infusions start at :05.
- Insulin rates come in 0.5-unit steps.
- Norepinephrine rates are continuous, and 32,919 records sit exactly at a cap of 1.00 mcg/kg/min. So ×2 pushes norepinephrine far outside the training range.
- Norepinephrine dose (mg) ≠ rate, so dose probably encodes body weight.
- glucose_finger averages 163 against glucose_lab 145, a systematic offset worth understanding.
- Arterial MAP exists in only 48% of stays.

Next I'll check:
- the vocab normalisation and bin edges (how doses are encoded);
- whether norepinephrine dose/rate is constant within a stay (weight);
- the finger-vs-lab difference at the same hour;
- the hourly pattern of readings and drugs per stay.

```python
V = json.load(open('/model/vocab.json'))
print(V.keys())
for k in V:
    if k not in ('value_bin_centres',):
        s = json.dumps(V[k]); print(k, s[:1500]); print()
print('centres', json.dumps(V.get('value_bin_centres'))[:2000])
ne = d[d.event=='norepinephrine'].copy()
ne['w'] = ne.dose/(ne.rate*60/1000)
print(ne.w.describe())
print('weight const within stay:', (ne.groupby('stay_id').w.std().fillna(0)<1e-6).mean())
```