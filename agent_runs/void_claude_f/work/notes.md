# NOTES (world-model dose repair)

## Setup facts (runs 1-5)
- Data: hourly grid. Readings at integer hours -1..24; infusion records start at h+0.0833 (5 min after the hour), run 1h.
  Stay: hospital_adm at -2, icu_adm at 0, ends 25.1667. 30k train stays, 400 tuning.
- 62% stays get insulin, 44% NE, 27% both. Insulin rate 0.5..18.5 step 0.5 (median 2); NE rate 0.01..1.0 (capped at 1.0),
  NE dose = rate*80kg*60/1000 (dose bin edges top at 4.8 -> rate 1.0).
- Shared tables: embed.bins.dose/rate (17x192), embed.numeric (192x6: z_value,has_value,z_dose,has_dose,z_rate,has_rate)
  are shared by insulin (code 8) and NE (code 9). Code-specific edits must go via encode_action nonlinearity or downstream.
- Doubling NE rate -> above 1.0 = out of training range (bin 16, z>0.77).

## Insulin policy & effect (runs 6-15)
- Insulin given iff glucose reading at hour h (finger, else lab) >= 150 (unrounded). Rate = base(G) * exp(eps), rounded to 0.5,
  base: [150,180)->1, [180,220)->2, [220,260)->3, [260,300)->4, >=300->6. eps iid N(0,~0.33): lag-1 autocorr 0.005,
  no stay component, uncorrelated with same-hour lab glucose given finger => as-if randomized dose variation.
- Regress future finger/lab glucose (lags 1..12h) on insulin rate, controlling G_h + step dummies: coef ~0 (+-0.04..0.1 mg/dL per U/h).
  => INSULIN RATE HAS ~NO CAUSAL EFFECT ON GLUCOSE (tentative). True x2, x1/2 effects ~ 0.

- RD at G=150 (insulin on/off, ~1 U/h): glucose jump at k=1..6h ~0 (+-0.3). Insulin inert at both margins.
## NE policy & effect (runs 17-25)
- NE trigger: arterial MAP in arterial stays (48.7% of stays have art line), cuff in cuff-only stays. Start if MAP<=65.
  Titration: rate_t = clip(rate_{t-1}*exp(step+eps), .., 1.0), rounded 0.01; step ~ +0.20 (MAP<=65), -0.01 (65-75], -0.26 (>75);
  eps sd ~0.3, iid (lag1 autocorr -0.03). Stops sometimes when MAP>75. No-MAP hours: step mean +0.077 (unclear what drives it).
- Effect: naive innovation regression gave small + (cap artefact). Clean subsample (rate_prev 0.1-0.4): innovation effect on
  next arterial MAP -0.08+-0.08, cuff +0.2+-0.17, later lags ~0 to -0.36. RD at 65 (NE start): no positive jump.
  => NE ALSO ~NO EFFECT ON MAP (tentative). All four true readouts ~0.
