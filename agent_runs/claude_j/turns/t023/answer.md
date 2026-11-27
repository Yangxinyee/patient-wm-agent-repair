Norepinephrine raises MAP, and the effect is short-lived. Per 1 mcg/kg/min for one infusion-hour, the fixed-dose impulse response on map_arterial is:

| hours after | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| change (mmHg) | +3.3 | +2.55 | +1.7 | +0.9 | +0.4 | +0.2 |

It is about 0 after that, with a cumulative +8.8 mmHg. Map_cuff gives the same curve, cumulative about +9.5, from mostly different stays. The shape is close to c·k·a^(k-1) with c = 3.3 and a ≈ 0.39, the same functional family as insulin (c = -2, a = 0.6). That points to a simulator in which each drug acts through a first-order compartment on a mean-reverting physiological variable.

Since I have about 22 runs of findings, I will now write them to the notebook before turning to the model. The next step after that is to see how the original model answers the dose questions.

```python
notes = r"""
# NOTES (world-model dose repair)

## Data facts (train.csv, 30k stays, hours -2..25.17; readings at integer hours, infusions at h+0.083 lasting 1h)
- Insulin: sliding scale on glucose_finger (GF) read 5 min before: GF<=150 none; (150,180]:1, (180,220]:2, (220,260]:3,
  (260,300]:4, >300:6 units/h, times iid lognormal multiplier (sd log ~0.31, mean rate per base 1.048/2.09/3.15/4.21/6.29),
  rounded to 0.5. Multiplier: no autocorrelation, uncorrelated with covariates -> exogenous instrument. dose==rate.
- GF measured every hour if last GF>150, else ~30% random. glucose_lab (GL) random 20% of hours. GF and GL same latent
  glucose (GF-GL mean 0, sd 9).
- NE: start iff map_arterial(MA)<65 (or MC if no MA) at h>=1 (hour-0 starts are partly random). Once on: titrated
  multiplicative random walk: log ratio step ~ +0.1 (MAP<65), ~-0.02 (65-75), ~-0.2..-0.3 (>75) + iid noise sd~0.29,
  rounded 0.01, capped 1.0; stops only when rate <~0.05. NE dose=4.8*rate. ~48% stays have arterial line (MA ~91%
  of hours, MAP-independent); others map_cuff (MC) ~86% of hours. MC-MA = +3 mmHg, sd 6.7.

## Causal effects (estimated from exogenous dose noise; RD at thresholds agrees for insulin)
- Insulin fixed-dose impulse response on glucose (per 1 unit/h for one infusion-hour started at h+0.083), at hour h+k:
  GF-based: [-2.00,-2.38,-2.08,-1.70,-1.16,-0.91,-0.65,-0.49,-0.25,-0.11,-0.09,0.01]; GL-based similar
  [-2.08,-2.45,-2.24,-1.52,-1.44,-1.02,-0.65,-0.43,...]. ~= -2*k*0.6^(k-1) (sum -12.5). Linear in dose, same across bands.
  RD at 150/180/220: one-hour effect -2.0/unit.
- NE impulse response on MAP (per 1 mcg/kg/min for one hour), MA-based: [3.3,2.55,1.69,0.92,0.37,0.21,-0.1,0,...]
  cum +8.8; MC-based similar (cum ~9.5). ~= 3.3*k*0.39^(k-1).
"""
open('/work/notes.md','w').write(notes)
print("saved")
```