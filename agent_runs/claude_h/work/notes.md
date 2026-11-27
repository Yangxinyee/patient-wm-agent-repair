# Notes

## Run 1-2: orientation
- Read ARCHITECTURE.md. Actions enter via encode_action -> u_k (to z_post and prelude). Dose edit changes insulin/norepi rate & dose z-scores, bins.

## Runs 3-13: data structure & insulin
- Hourly grid: readings at integer hours -1..24, drug records at h+0.0833 (run 1h). Stays end 25.17.
- Insulin policy: sliding scale on finger glucose (lab if no finger): base 0 (<150),1 (150-180),2 (180-220),3 (220-260),4 (260-300),6 (>=300),
  times iid multiplicative jitter (0.5..2, median 1; autocorr 0.005, no stay effect, uncorrelated with trend); ~3% omissions. No glucose reading -> no insulin.
  If insulin at t-1 -> finger reading at t always present (else 31%).
- Panel saved /work/panel_train.pkl, /work/panel_tune.pkl.
- KEY: with insulin>0 sample & 5mg/dL glucose-bin FE, jitter-identified effect of rate on G(t+k)-G(t), k=1..8: all ~0 (+0.001,-0.03,+0.06,+0.04,+0.08,+0.01,+0.11,-0.02; se .04-.08 mg/dL per u/h).
  Naive association is negative (-0.75) from regression to mean. => Insulin appears to have ~NO effect on finger glucose. (to verify: nonlinearity, lab)

## Runs 14-28: norepinephrine; model baseline
- NE policy: titration. Start if MAP<~64.5 (rate higher if MAP lower); continuing: rate_t = g(MAP_t, rate_{t-1}) + iid innovation (sd .138, autocorr -.001,
  uncorrelated w/ MAP trend, cuff reading, prior MAPs). Shock persists in later rates R=.67,.48,.35,...
- NE LP of MAP_{t+k}-MAP_t on innovation: k=1 +0.14(.09), later -0.05..-0.6 (se .1-.25). Summary avg6-12 -0.28(.14), cuff -0.29(.13); hours<12 -0.36(.16), >=12 +0.05(.11). => weak/inconsistent; treat as ~0.
- Insulin: LP k=1..12 all ~0 (se .04-.11), RD at 150/180/220/260 ~0, by subgroup ~0. => insulin effect ABSENT.
- FWD 1 (dose_edit original, tuning sel1 100 stays, /work/sel1.json): ins x2 +0.066, x1/2 -0.032, ne x2 -0.027, ne x1/2 +0.009 (all ~0; stay-clustered se ~.04/.02).
  x2 and x1/2 shifts correlate +0.58 per stay-hour => partly perturbation noise.
- Current belief: true effects ~0 for both drugs; original model already ~0. Option: make model exactly dose-invariant (shift=0) if NLL holds.

## Runs 29-36
- NE innovation does not predict next MAP in no-MAP hours (-0.22 se .55): doses carry no hidden info.
- Implied NE x2 readout via LP deconvolution (h_k) applied to tuning reached rows: arterial -0.31 (boot sd .15); cuff-only stays (independent) -0.37 (boot mean -0.28 sd .36).
  Pooled ~ -0.32 +- 0.14. Insulin same method: -0.04 (boot sd .49). Stops are mechanical (only when rate<=0.1) -> not usable.
- Belief: insulin ABSENT (true 0). NE: most likely absent; maybe tiny delayed negative (x2 ~ -0.3, x1/2 ~ +0.15).
- Plan B: make model dose-invariant (zero z_dose/z_rate cols, constant bin rows, code-specific mean compensation) => readouts exactly 0. Check NLL.
- Run 37: kernel OOM (don't materialize 300k x 192 arrays).

## Runs 39-43 (after restart 1)
- Current edit (Plan B, dose-invariant): embed.bins.dose & embed.bins.rate rows 1..16 := original row 8 (median bin); row 0 untouched.
  embed.numeric.weight cols z_dose (2) and z_rate (4) zeroed. Check: drug-atom embedding term identical for x1, x2, x1/2 (dev 1.5e-8).
- FWD 2 factual both, tuning sel1 (100 stays, 5323 values): original 1.34364, edited 1.34369 (+0.00005). By event all within +-0.0005. => NLL SAFE.
- Next: sharpen NE evidence (variance-reduced LP with richer covariates), then dose_edit both to confirm edited readouts = 0.

## Runs 44-54: sharpened effect estimates (cross-fitted HGB residualization, LP + deconvolution, stay bootstrap)
- NE (innov sd .136, mean rate .57): implied x2 readout arterial -0.24(.14), cuff -0.47(.18), combined MAP -0.34(.14).
  Spec sensitivity: minimal controls -1.05 vs full -0.40 -> caused by 12k 'no-MAP-at-t' rows (titrated on last MAP, age 1).
  On rows with a MAP reading at t: full -0.34(.13), minimal -0.20(.15); t>=12 -0.18(.19). Lag1 ~0. No-hidden-info: policy uses arterial
  if present (cuff coef ~0 given art), equal slopes art/cuff (no dilution). Policy: step (up ~.03 if MAP<65, hold 65-74, down if >75), cap ~1.0.
- Insulin: rates rounded to 0.5 u/h ('exact' jitter share depends on base -> U-shape artifact explained). Jitter-only implied x2 readout
  finger +0.90(.51) (stable min vs full spec: +0.99), lab +0.30(.62); lag1 +0.03(.04). Omissions -> slightly lower later glucose (noisy).
  Placebos: insulin->MAP 0 (se .02). Presence of later readings independent of innovations (no selection artifact).
- Interpretation: any designer-plausible 'small' effect (few mg/dL or mmHg, either sign) would be 6-18 sigma; observed are 1.5-2.6 sigma
  and fragile => treat both as ABSENT. Keep dose-invariant edit (exact 0 readouts). Next: confirm with dose_edit both on new stays.

## Runs 55-63 (after restart 2)
- FWD 3 (dose_edit both, tuning 100 stays; 39 insulin stays/397 stay-hours, 5 NE stays/56 stay-hours): original ins x2 +0.052, x1/2 -0.152,
  ne x2 +0.008, ne x1/2 -0.023; EDITED all shifts exactly 0.0 at every stay-hour (shift_se 0).
- FWD 4 (factual both, 100 other tuning stays, 4529 values): original 1.58471, edited 1.58464 (-0.00007). NLL SAFE (2 checks).
- Checking ARCHITECTURE for other dose paths (last_numeric z_last_rate, last rate bin) before finalizing.

## Final (runs 64-65)
- ARCHITECTURE: dose values enter only via drug-atom embeddings (bins.dose, bins.rate, numeric z_dose/z_rate) -> encode_action -> u_k,
  and via the per-code 'last' register (z_last_rate, last rate bin) which for glucose/MAP codes never holds a rate (only drug-attribute
  heads use it; not used in rollout or value NLL). Edit neutralizes path 1 => reading predictions exactly dose-invariant on any stay.
- Final belief: insulin effect on glucose ABSENT (jitter LP ~0, RD at sliding-scale thresholds ~0, implied x2 +0.9+-0.5);
  NE effect on MAP ABSENT or negligible (implied x2 -0.34+-0.14, spec-fragile, t>=12 ~0); policies depend on recorded readings
  only (sharp thresholds, no art/cuff dilution) so jitter identification is clean. Expected true readouts ~0 for all four.
- SUBMIT dose-invariant edit: readouts exactly 0 (FWD 3), NLL unchanged (FWD 2: +0.00005, FWD 4: -0.00007).
