The records have a regular structure. Glucose_finger and map_cuff are read on the hour. Insulin infusion records start about 5 minutes after the hour (hour + 0.0833) and run for one hour.

Insulin rates are 0.5–16 units/hour, typically 1–2. Norepinephrine rates are 0.01–1.0 mcg/kg/min. For norepinephrine, dose divided by rate is about 4.8, which suggests a fixed weight of about 80 kg, but I need to check that.

Next I'll look at:
- the vocab: per-code normalisation and bin edges, since the dose edit acts through z-scores and bins;
- whether a doubled dose falls outside the training range, which would matter for how the model extrapolates;
- how rates evolve within a stay (step sizes, dependence on the glucose just read), to find the dosing policy;
- dose/rate ratios.

```python
print(json.dumps(V['codes']))
print(json.dumps(V['event_kinds']), json.dumps(V['atom_kinds']))
print(V['numeric_note'])
print(V['value_bin_note'])
for k,v in V['numeric'].items():
    print(k, json.dumps(v)[:600])
print(json.dumps(V['value_bin_centres'])[:1500])
```