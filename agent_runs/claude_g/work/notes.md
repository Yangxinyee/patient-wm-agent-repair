# Notebook

## Session 1
- Read ARCHITECTURE.md and README. Dose enters only via encode_action (u_k) -> z_post, prelude.

## Data findings (session 1)
- All stays 25.17 h; readings on integer hours; infusions start h+0.0833 and last 1 h.
- Insulin policy: base by latest glucose reading (finger, else lab): >150:1, >180:2, >220:3, >260:4, >300:6 u/h, times
  iid multiplier m (lognormal-ish, sd~1/3), rounded to 0.5. m uncorrelated with everything (lag1 corr 0.002, no stay effect).
- NE policy: start iff MAP<65 (sharp), titrate ratio depends on MAP (<65 up, 65-75 flat, >75 down ~0.8), cap 1.0.
- CAUSAL: insulin dose-margin effect on glucose_finger ~0 at lags 1..12 (coef ~ +0.05+-0.05 mg/dL per u/h).
  RD at 150 threshold (start insulin): no jump. => INSULIN INERT.
- NE: DML margin effect on map_arterial -0.33+-0.08 mmHg per mcg/kg/min at lag1, ~0 later; RD at MAP 65 start: no jump.
  => NE INERT (at most -0.2 mmHg for x2).

## Model baseline (fwd_1, tuning set100): original dose_edit readouts: ins x2 +0.114, ins x1/2 -0.148, ne x2 +0.005, ne x1/2 -0.006
  -> original already ~0 (matches 'inert'). Extra checks: RD at all insulin thresholds 150..300: no jumps. Richer controls: insulin coef ~0.01+-0.04.
## Edit A (dose-invariance): embed.bins.dose & embed.bins.rate rows1..16 := row 8 (median bin both drugs), embed.numeric cols 2,4 (z_dose,z_rate) := 0.
  Makes every drug atom look like the median dose of its drug -> dose edits give exactly 0. Measurement atoms untouched.

- fwd_2 factual both rand100 tuning: NLL orig 1.49270 vs edited(A) 1.49271 (+0.00001). fwd_3 dose_edit edited(A): all four readouts exactly 0.000.
- NE DML lags: -0.35(1),-0.38(2),-0.27(3),-0.13(4),-0.12(6),+0.06(8),+0.25(10),+0.58(12) per mcg/kg/min; NEGATIVE CONTROL (cuff MAP at h, pre-dose)
  gives -0.54+-0.21 -> titration tracks latent MAP -> short-lag negative is confounding. NE effect ~0 (implied |readout| < 0.5 mmHg).
- DML insulin k=1: 0.006+-0.038. BELIEF: both drugs inert; target readouts all 0; Edit A achieves exactly 0 with intact NLL.

- Cumulative randomization test (insulin, hours 13-24 vs summed random dose parts since 12): slope 0.002+-0.065/unit -> implied x2 readout +0.03+-0.93 mg/dL.
- fwd_4 factual both on drug-heavy set100: orig 1.34164 vs edited(A) 1.34146 (-0.00018). Edit A safe.
## FINAL DECISION: submit Edit A (dose-invariance). Expected readouts: all four exactly 0.0; true effects estimated ~0
   (insulin x2 ~0+-1 mg/dL, x1/2 ~0+-0.5; NE x2/x1/2 within +-0.5 mmHg, best estimate 0).
