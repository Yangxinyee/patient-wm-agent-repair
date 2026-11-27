The norepinephrine rule is titration from the previous rate.
- **MAP below 65:** the rate goes up, by about +0.07 to +0.15 on the log scale on average.
- **MAP 65–75:** the rate is held.
- **MAP above 75:** the rate comes down by about 0.2–0.28 on the log scale.

There is large multiplicative noise (log-ratio sd ≈0.29) and a cap at 1.0. Arterial MAP is used when present, cuff otherwise, with the same thresholds. As with insulin, there is random dose variation plus sharp thresholds (65, 75) usable for discontinuity checks.

Next I'll record these findings in the notebook, then study glucose dynamics. The first effect estimate compares next-hour finger glucose against the insulin rate within narrow bands of current glucose. That uses the dose noise as the source of variation. Later I'll cross-check it against the 180/220 discontinuities.

```python
notes = """
## Runs 3-10: data structure & dosing policy (train.csv)
- All stays 25.17h. Readings at integer hours (-1..24); infusions at h+0.0833, each 1h. Insulin dose==rate; NE dose=4.8*rate.
- Glucose finger/lab intermittent; map_arterial hourly in ~48% stays (art line); map_cuff ~60% hours.
- INSULIN POLICY (uses the reading taken that hour, finger; lab if no finger): ON if G>160 (always); 140-160 depends on prev on (0.68 vs 0.23); <140 ~off.
  Base rate steps on recorded finger value: ~1.05 (<180), ~2.08 (180-220), ~3.15 (220-260?), higher above; sharp jump AT the recorded value
  (at exactly 180 mean 1.56 -> threshold on unrounded reading). Multiplicative noise, rounded to 0.5 (sd ~0.37 at base1, ~0.65 at base2).
- NE POLICY: start if MAP<65 (art if available else cuff); titrate: MAP<65 up (log ratio +0.07..0.15), 65-75 hold, >75 down (-0.2..-0.28);
  log-ratio noise sd ~0.29; cap 1.0. Thresholds sharp at 65/75.
- IDEA: dose noise given recorded reading = natural experiment; thresholds (180,220; 65,75) allow regression discontinuity (RD) check vs hidden confounding.
"""
with open('/work/notes.md','a') as f: f.write(notes)

# glucose dynamics: next-hour finger vs current finger and insulin rate
P['g_next'] = P.groupby('stay_id').glucose_finger.shift(-1)
P['gl_next'] = P.groupby('stay_id').glucose_lab.shift(-1)
P['ins0'] = P.insulin.fillna(0)
A = P[(P.h>=0)&(P.h<24)&P.glucose_finger.notna()&P.g_next.notna()].copy()
A['gb'] = pd.cut(A.glucose_finger,[0,140,150,160,170,180,190,200,220,240,260,300,1000])
# within-bin regression of g_next on ins0
import statsmodels.api as sm  # may not exist
```