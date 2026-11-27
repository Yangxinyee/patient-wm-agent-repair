Norepinephrine raises arterial MAP. Per 1 mcg/kg/min for one hour, with later doses held fixed:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|
| ΔMAP (mmHg) | +3.2 | +2.8 | +1.6 | +0.8 | +0.6 | ≈0 |

The cumulative effect is about +8.6 mmHg per sustained 1 mcg/kg/min. Cuff MAP gives a consistent +8.1, which is an independent check. With typical rates around 0.5, doubling from hour 12 should raise MAP by roughly +3 to +4 mmHg once established. The response is faster than insulin's: it peaks at lag 1–2 and is gone by about lag 6.

Next I'll compute the implied true readouts for the tuning stays, assuming linear superposition of these per-dose effects over the recorded doses from hour 12. I'll then run one dose_edit forward pass on 100 tuning stays with both models to see:
- the output format and exactly how "hours after the drug had run" is defined;
- what the model currently predicts.

Comparing the two tells me how wrong the model is. Here I list tuning stays by drug exposure after hour 12 so I can pick an informative set.

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Runs 18-23: norepinephrine
- NE POLICY: rate_t = clip((rate_{t-1} + D(MAP_t)) * exp(eta), 0.01, 1.0); eta sd~0.3 multiplicative.
  D: MAP<55 +0.08; 55-65 +0.04; 65-75 0; >75 -0.04/-0.05. Arterial used if present, else cuff.
- residual (rate - cell mean given prev rate x MAP cat) exogenous: all corr ~0.
- g-estimation NE -> MAP art per 1 mcg/kg/min: [3.18,2.81,1.58,0.80,0.57,0.03,0.14,...~0], cum +8.6 (cuff +8.1).
""")
PT = PT.sort_values(['stay_id','h']).reset_index(drop=True)
tu_ins12 = PT[(PT.h>=12)&PT.insulin.notna()].groupby('stay_id').size()
tu_ne12 = PT[(PT.h>=12)&PT.norepinephrine.notna()].groupby('stay_id').size()
tstays = PT.stay_id.unique()
print("tuning stays", len(tstays), "with insulin>=12:", len(tu_ins12), " with ne>=12:", len(tu_ne12), " both:", len(set(tu_ins12.index)&set(tu_ne12.index)))
tu_art = PT[(PT.h>=13)&PT.map_arterial.notna()].groupby('stay_id').size()
print("ne>=12 stays with arterial after 13:", len(set(tu_ne12.index)&set(tu_art.index)))
print(tstays[:5])
```