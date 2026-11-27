## Step 3: how dose values map to model inputs, and how clinicians dose

**What the last result showed.**
- Insulin rate and dose are numerically identical (dose = rate × 1 h). Both have median 1.5 and IQR 1.0, and the bin edges are on a 0.5-step grid up to 5.
- Norepinephrine dose = rate × 4.8 (an 80 kg patient over 60 minutes), with median rate 0.47 and IQR 0.6.
- The rate bin edges pile up at 1.0, so rates are probably capped at 1.0 mcg/kg/min. Doubling a capped rate (to 2.0) would put the model outside its training range in both z and bins. Extrapolation behaviour will therefore matter.

**What I will do next.** Characterise the dosing:
- the distinct rate values and how often each occurs;
- stays per drug, and when infusions start and stop;
- how the rate relates to the reading just before it.

**What this tells me.** Whether dosing is a deterministic function of the last reading (a protocol), which would give discontinuities, or noisy, which would leave a random component. I will also check for an apparent cap (rates at 1.0) and the readings' measurement cadence.

```python
ins = tr[tr.event=='insulin']; ne = tr[tr.event=='norepinephrine']
print("insulin rate counts:\n", ins.rate.value_counts().sort_index().to_string())
print("NE rate quantiles:", ne.rate.quantile([0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).to_dict())
print("NE rate top values:\n", ne.rate.round(4).value_counts().head(30).to_string())
print("NE dose/rate ratio:", (ne.dose/ne.rate).describe())
print("insulin dose/rate:", (ins.dose/ins.rate).describe())
print("stays with insulin:", ins.stay_id.nunique(), " with NE:", ne.stay_id.nunique())
L = tr[tr.event=='icu_admission'][['stay_id','end_hour']].set_index('stay_id').end_hour
print("ICU LOS quantiles:", L.quantile([0,.1,.25,.5,.75,.9,1]).to_dict())
print("insulin hour frac:", (ins.hour%1).round(4).value_counts().head().to_dict())
print("NE hour frac:", (ne.hour%1).round(4).value_counts().head().to_dict())
for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
    x = tr[tr.event==ev]
    print(ev, "n stays", x.stay_id.nunique(), "hour frac", (x.hour%1).round(4).value_counts().head(3).to_dict(), "value q", x.value.quantile([0,.05,.5,.95,1]).to_dict())
```