# Weight-repair notebook
## Initial plan
- Inspect architecture and data before choosing an edit.
- Infer drug effects solely from the supplied simulated-ICU records.
- Distinguish treatment-selection associations from treatment effects.
- Use statistical fits and algebraic weight edits only; no gradients or optimiser.
- Validate factual value NLL and rollout effects, including cross-sensor consistency.
- No findings or edits yet.

## Inspection
- Architecture: drug action embeddings affect the recurrent state shared by all heads.
- Value heads also have code-specific last-reading transitions.
- Broad action edits may affect all-output log loss; prefer selective, evidence-based edits.
- Python runs: 2. Forward passes: 0. No edits yet.

## Hourly panel and preliminary associations
- Infusions begin at hour t+0.0833, after hour-t measurements.
        sensor          controls      n   insulin  insulin_se  norepinephrine  norepinephrine_se  residual_sd
glucose_finger  current readings 298450 -2.098636    0.042771       -0.243935           0.140247    13.540091
glucose_finger full past history 298450 -2.169121    0.039892       -0.064443           0.280626    12.713684
   glucose_lab  current readings  28483 -1.108132    0.095744        0.520534           0.234678     7.135888
   glucose_lab full past history  28483 -1.475258    0.104063        0.347124           0.488524     7.053169
  map_arterial  current readings 312563  0.021053    0.016460       -0.616412           0.050659     4.727718
  map_arterial full past history 312563 -0.001370    0.016657        1.982264           0.085269     4.380650
      map_cuff  current readings 333382 -0.035490    0.031285       -6.659195           0.153377     7.988392
      map_cuff full past history 333382 -0.035838    0.027808       -0.664094           0.218828     7.302487
- These are conditional associations, not established causal effects.
- Python runs: 3. Forward passes: 0. No edits yet.

## Baseline rollout and flexible residualisation
- Original tuning first 100: insulin x2 -3.68; half +2.018; NE x2 +0.465; half -0.871.
- Two-fold cross-fitting by stay; flexible history controls and treatment-presence indicators.
        sensor                  sample      n   insulin  insulin_se  norepinephrine  norepinephrine_se
glucose_finger       all next readings 400154 -2.009795    0.045709        0.385314           0.264109
glucose_finger current sensor observed 298450 -2.018225    0.046122        0.520053           0.302609
   glucose_lab       all next readings 143791 -2.090995    0.076287       -0.342455           0.364291
   glucose_lab current sensor observed  28483 -1.846260    0.158770       -0.031431           0.581099
  map_arterial       all next readings 328940  0.008857    0.021188        3.121313           0.089562
  map_arterial current sensor observed 312563  0.001712    0.021342        3.150740           0.090147
      map_cuff       all next readings 439865 -0.049094    0.029712        2.723798           0.176977
      map_cuff current sensor observed 333382 -0.044717    0.033971        2.787206           0.228032
- Residualised estimates still require overlap, placebo and stability checks.
- Python runs: 4. Forward passes: 1. No edits yet.

## Factual baseline and latent dynamics
- Original factual first 100 tuning: value NLL 1.63164, 4,751 readings.
          drug      rho      beta  process_sd  baseline_walk_sd  sensor1_sd  sensor2_sd  sensor2_offset      nll  success   seconds
       insulin 0.967881 -1.272755    4.758194          0.205145    8.921139    3.537458       -0.042686 3.852165    False 10.802939
norepinephrine 0.900838  0.295787    1.766206          0.241436    3.263101    6.086621        3.147190 3.188606     True  4.784169
          drug  factor  mean_shift_all400  mean_shift_first100
       insulin     0.0          10.068514             9.670204
       insulin     0.5           5.034257             4.835102
       insulin     1.5          -5.034257            -4.835102
       insulin     2.0         -10.068514            -9.670204
       insulin     3.0         -20.137027           -19.340408
norepinephrine     0.0          -0.692790            -0.655035
norepinephrine     0.5          -0.346395            -0.327518
norepinephrine     1.5           0.346395             0.327518
norepinephrine     2.0           0.692790             0.655035
norepinephrine     3.0           1.385581             1.310070
- Targets above are provisional statistical estimates, not simulator ground truth.
- Python runs: 5. Forward passes: 2. No edits yet.

## Causal identification checks
- Paired-sensor IV suggests insulin lowers next glucose, approximately 2 per rate unit.
- Norepinephrine sensor-difference instrument is weak; do not use its IV ratios.
- Earlier zero placebos were mechanical because controls contained the placebo outcome; corrected here.
          drug     n       beta        se  dose_slope  past_outcome_slope
       insulin 59658  -0.104668  0.609526    0.025015           -0.002618
norepinephrine 93508 -20.380205 14.355973   -0.000347            0.007077
          drug  maxlag      n       rho   rho_se      beta  beta_se
       insulin       2 224413 -2.849041 0.509204 18.162388 3.203651
       insulin       3 202551  0.397434 0.149430 -2.194095 0.960694
       insulin       4 184978  0.918568 0.109879 -5.462461 0.726760
norepinephrine       2 270360  0.529543 0.037465  2.555445 0.294576
norepinephrine       3 245698  0.539414 0.026764  2.774850 0.272898
norepinephrine       4 222850  0.513011 0.025379  2.627387 0.291069
- Python runs: 9. Forward passes: 3. No edits yet.

## Dose-innovation local projections
- Cross-fitted policy uses current/past sensor information and past doses; no future features.
- These responses include subsequent clinician-policy feedback, unlike fixed-dose evaluation.
          drug  horizon      n  response       se
       insulin        1 400154 -1.917161 0.045098
       insulin        2 384677 -2.199973 0.051051
       insulin        4 353362 -1.274455 0.059664
       insulin        8 290018 -0.117182 0.078432
norepinephrine        1 328940  3.038176 0.089059
norepinephrine        2 315253  4.739887 0.106138
norepinephrine        4 287837  5.065232 0.130097
norepinephrine        8 233088  2.784013 0.173857
- Dynamic-panel insulin estimates unstable; not suitable as repair targets.
- Python runs: 10. Forward passes: 3. No edits yet.

## Sensor-proxy dynamic check
- Instrument current sensor with its paired sensor, controlling only earlier history.
- Additional assumption: dose is independent of regressor sensor measurement noise; test sensor-swap stability.
          drug  swap     n      rho   rho_se      beta  beta_se
       insulin False 59611 1.152056 0.008436 -7.162353 0.123087
       insulin  True 59611 1.000964 0.007962 -1.787155 0.082103
norepinephrine False 93413 0.881754 0.010005  4.200352 0.171239
norepinephrine  True 93413 0.899401 0.009205  3.228365 0.235937
- Python runs: 11. Forward passes: 3. No edits yet.

## Fixed-dose response-kernel estimates
- Fit geometric kernels to instrument/outcome moments, including subsequent recorded doses.
- Insulin instrument: paired-sensor noise difference with past-only controls.
- NE instrument: cross-fitted policy innovation with measured-history controls; stronger sequential-unconfoundedness assumption.
- insulin: rho=0.82939, beta=-2.30803
- norepinephrine: rho=0.63511, beta=3.55456
          drug  factor   first100     all400
       insulin     0.0  10.528802  11.082850
       insulin     0.5   5.264401   5.541425
       insulin     1.5  -5.264401  -5.541425
       insulin     2.0 -10.528802 -11.082850
       insulin     3.0 -21.057603 -22.165700
norepinephrine     0.0  -3.808429  -3.969327
norepinephrine     0.5  -1.904215  -1.984664
norepinephrine     1.5   1.904215   1.984664
norepinephrine     2.0   3.808429   3.969327
norepinephrine     3.0   7.616858   7.938655
- Python runs: 12. Forward passes: 3. No edits yet.

## Head reconstruction setup
- Original factual training first 100: value NLL 1.64546 (4,510 readings).
- Saved states matched to preceding bundles; testing numerical head conventions before any edit.
- Insulin long-horizon kernel remains weakly identified; NE evidence is more stable.
- Python runs: 16. Forward passes: 4. No edits.

## Reconstruction diagnostics
- Best initial conventions: GELU, type 1, interleaved last numeric, status 0, seconds/10.
- Reconstruction scaled mean RMSE ~0.115; expected four-prior-draw noise ~0.0475.
- Bin boundary, centre-value, and latent averaging checks did not explain the remaining error.
- Performing a bounded query/context convention check; no edits.
- Python runs: 19. Forward passes: 4.

## Restart diagnostics
- Runs 20–22: recovering head helpers and checking saved-state chronology and register dimensions. No weight edits; four forward passes used.

- Run 23: saved out equals layer-normalized mean h to 7.3e-8 RMSE. Current helpers use seconds-based flow, log1p(seconds)/10, interleaved last-number fields and measurement query type 1. These still give scaled-mean RMSE 0.114; no edits.

- Run 24 and forward 5: chronology is exact; absent-bin and status embeddings add zero here. Tested separate prior/world contexts and latent packing without resolving reconstruction error. Repeated original factual pass on 25 reordered training stays to quantify noise. No edits.

- Run 25: repeated factual predictions differ by scaled mean RMSE 0.07198 and NLL RMSE 0.24980, implying single-pass sampling noise ~0.051. Reconstruction has larger systematic positive bias. All matched state IDs are correct. Insulin kernel bootstrap is very weak (rho often reaches bounds); NE kernel is substantially more stable. Treat long-horizon insulin targets as uncertain. No edits; five forward passes.

- Run 26: fresh reconstruction exactly reproduced existing inputs. Documented flow still gives scaled mean RMSE 0.114 and bias 0.054; doubled flow improves RMSE to 0.097 but is not yet justified. No edits. Re-reading architecture and recovering kernel context before proceeding.

- Runs 27–32: fresh prior scales are ordinary; latent averaging does not explain head bias. World-view coefficient checks still favor documented terms, though double flow reduces error. Exhaustively checking the undocumented extra last_numeric fields before any edit. No edits; five forward passes.

- Run 32: moving the last-value input to another column lowers mean RMSE but worsens NLL RMSE to 0.51; rejected as an unjustified convention. Returning to direct original factual residuals and dose innovations for evidence-based repair targets. No edits.

## Runs 34–36: restart recovery
- Re-read architecture and record schema. Original first-100 tuning rollouts: insulin x2 -3.680, half +2.018; NE x2 +0.465, half -0.871.
- Proposed repair: value-head exponential tilt driven by decoded dose-policy innovations, with sensor gating and held-out checks.
- Insulin long-horizon causal target remains uncertain. No edits; 5 forward passes used.

## Run 37: dose-feature feasibility
- Reconstructed deterministic hidden features for all 4,510 training readings; tested two-fold-by-stay ridge decoding of sensor-gated dose-policy innovations and raw preceding doses.
           target   alpha           drug    n  r2_relevant  target_sd  prediction_sd  off_sensor_rms
policy_innovation    0.01        insulin 1696    -0.126018   0.354702       0.143518        0.067941
policy_innovation    0.01 norepinephrine 2481    -0.071898   0.071507       0.025139        0.016906
policy_innovation    1.00        insulin 1696    -0.110218   0.354702       0.134487        0.061409
policy_innovation    1.00 norepinephrine 2481    -0.058460   0.071507       0.022979        0.014955
policy_innovation   10.00        insulin 1696    -0.065987   0.354702       0.107268        0.044138
policy_innovation   10.00 norepinephrine 2481    -0.050982   0.071507       0.020018        0.010454
policy_innovation  100.00        insulin 1696    -0.027232   0.354702       0.075224        0.023121
policy_innovation  100.00 norepinephrine 2481    -0.037199   0.071507       0.015383        0.005760
policy_innovation 1000.00        insulin 1696    -0.008007   0.354702       0.053705        0.009840
policy_innovation 1000.00 norepinephrine 2481    -0.020083   0.071507       0.009394        0.003161
         raw_dose    0.01        insulin 1696     0.779093   0.982529       0.875262        0.108713
         raw_dose    0.01 norepinephrine 2481     0.679226   0.283452       0.250746        0.051777
         raw_dose    1.00        insulin 1696     0.782466   0.982529       0.872275        0.101533
         raw_dose    1.00 norepinephrine 2481     0.679420   0.283452       0.248594        0.047104
         raw_dose   10.00        insulin 1696     0.789846   0.982529       0.855458        0.088268
         raw_dose   10.00 norepinephrine 2481     0.685891   0.283452       0.244065        0.036642
         raw_dose  100.00        insulin 1696     0.787012   0.982529       0.823658        0.078820
         raw_dose  100.00 norepinephrine 2481     0.696316   0.283452       0.238938        0.025575
         raw_dose 1000.00        insulin 1696     0.740943   0.982529       0.740144        0.074933
         raw_dose 1000.00 norepinephrine 2481     0.700982   0.283452       0.225333        0.025580
- No edits yet. Five forward passes used.

## Run 38: readout vs state information
- Sensor-gated hidden-feature innovation decoding failed on held-out stays despite good raw-dose decoding (R2 ~0.79 insulin, ~0.70 NE).
- Tested innovation decoding earlier in the state, separately by relevant sensor family.
    features           drug  alpha        r2  correlation
   all_slots        insulin   10.0  0.028877     0.266526
   all_slots norepinephrine 1000.0 -0.012698     0.023955
     context        insulin   10.0  0.011676     0.188526
     context norepinephrine 1000.0 -0.009584     0.007722
      mean_h        insulin   10.0  0.018817     0.206398
      mean_h norepinephrine 1000.0 -0.008433     0.015250
         out        insulin   10.0  0.009317     0.187724
         out norepinephrine 1000.0 -0.009127     0.010135
 posterior_z        insulin   10.0  0.104526     0.326598
 posterior_z norepinephrine   10.0  0.084331     0.295506
value_hidden        insulin 1000.0 -0.010345     0.055903
value_hidden norepinephrine 1000.0 -0.062948    -0.017338
  world_view        insulin   10.0  0.010615     0.188121
  world_view norepinephrine 1000.0 -0.011510     0.001111
- No edits; five forward passes used.

## Run 39: first reversible weight probe
- Posterior z retains some innovation signal (held-out R2 ~0.10 insulin, ~0.08 NE); value hidden does not. This supports probing action transmission rather than fitting an innovation readout to unsupported features.
- Temporarily doubled embed.numeric.weight columns 2 (z_dose) and 4 (z_rate), leaving measurement numerical inputs untouched.
- Next: paired original/edited factual validation on first 100 tuning stays, followed by rollout if useful. This is a probe, not an accepted repair. Five forward passes used.

## Forward 6 and run 40
- Doubling embed.numeric.weight dose/rate columns: tuning first-100 original value NLL 1.63164; edited 1.63216; difference +0.00052 (4,751 readings), below +0.005 tolerance.
- This establishes only factual feasibility, not causal improvement. Recovering exact IDs and identification evidence before paired rollouts.
- Python runs: 40. Forward passes: 6. Probe edit remains active.

- Run 41: bounded recovery after run 40 output truncation. Architecture confirms the action-input probe affects shared state; all-head effects remain a concern. No further edits.

## Forward 7 and run 42: rejected input-gain probe
- Original/edited insulin x2: -3.680/-3.638; half: +2.018/+2.380. NE x2: +0.465/+0.342; half: -0.871/-1.090.
- Scaling numerical dose/rate columns did not repair the main response and weakened NE doubling. Restored original embed.numeric.weight.
- Next: investigate selective algebraic readout/state corrections, with factual preservation and dose sensitivity checked separately.
- Python runs: 42. Forward passes: 7. No active edits.

- Run 43: inspecting latent/action pathways and cached helpers for an innovation-based algebraic repair. No edits. Any candidate needs held-out factual and intervention checks.

- Run 44: collecting exact embedding conventions to validate posterior reconstruction before any latent-path edit. No edits; seven forward passes used.

- Run 45: single-action posterior reconstruction convention test; no edits.
  - insulin: posterior RMSE 2.101383e-05, configuration ('left', 10.0, 'encoded_empty', (np.int64(3), np.int64(2), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0))).
  - norepinephrine: posterior RMSE 2.254109e-05, configuration ('left', 10.0, 'encoded_empty', (np.int64(4), np.int64(3), np.int64(0), np.int64(0), np.int64(0), np.int64(0), np.int64(0))).

- Run 46: validated END/measurement conventions before latent repair.
  - END results: {'insulin': (0.0702604129910469, ('absent', 'observation', (np.int64(1), np.int64(1), np.int64(0), np.int64(1), np.int64(1), np.int64(0), np.int64(1)))), 'norepinephrine': (0.046266671270132065, ('original', 'observation', (np.int64(5), np.int64(2), np.int64(0), np.int64(0), np.int64(1), np.int64(3), np.int64(1))))}.
  - Best measurement-unit result: (0.049574967473745346, 1, 2).

## Restart recovery, runs 47–48
- Re-read architecture and schema. No active edits. Action-only posterior reconstruction is exact, but general bundle reconstruction is not yet trusted. Recovering causal estimates and cached helpers before choosing a selective repair.

## Run 49: reversible dose-bin gain probe
- Doubled non-absent dose/rate bin embeddings, leaving numerical inputs and absent rows unchanged. This is an exploratory probe, not an accepted causal repair. Next require paired factual NLL and rollout validation. Seven forward passes used.

## Forward passes 8–9; run 50: rejected bin gain
- Bin gain factual NLL +0.00029, but insulin x2/half changed from -3.680/+2.018 to -3.198/+1.436; NE x2/half from +0.465/-0.871 to +0.552/-0.642. Restored both tensors.
- Simple input gains have failed. Next investigate a history-conditioned action innovation in the posterior pathway.
- Nine forward passes used. No active edits.

## Run 53: restart recovery
- Drug-only posterior reconstruction is exact; observation/END reconstruction remains unvalidated.
- Recovering architecture, identification evidence, and cached helpers before another reversible edit.
- No active edits; nine forward passes used.

## Run 57: posterior counterfactual features
- Held factual context fixed and computed action-encoder/posterior hidden features for factors 0, .5, 1.5, 2, 3.
- Next test: a cross-stay regression with low factual activation and dose-proportional counterfactual differences.
- No weight edits; nine forward passes used.

## Runs 58–59: posterior dose-contrast feasibility
- First fit failed on nonfinite reconstructed boundary contexts; excluded invalid rows without imputation.
- Re-ran two-fold-by-stay decoder fits and saved posterior_contrast_cv.csv.
- Computed original latent dose directions from counterfactual contrasts.
- No weight edits; nine forward passes used.

## Run 64: reversible posterior-only probe
- Added rank-two dose-contrast corrections to z_post.2 mean rows and corresponding biases; log-SD rows unchanged.
- Strength 4 for both drugs, using innovation decoders with factual_weight=10 and alpha=100 and original latent dose directions.
- This is a pathway probe, not an accepted repair. Require paired held-out factual fit and rollout response before retaining it.
- Python runs: 64. Forward passes: 9.

## Forward 10–11; run 65: rejected posterior probe
- Factual NLL change -0.00003, but rollout changes negligible: insulin -3.665/+2.041; NE +0.472/-0.837.
- Restored both z_post tensors. Posterior amplification is not an effective repair at this strength.
- Next inspect direct action state transmission and dose-contrast features at the value readout.
- Python runs: 65. Forward passes: 11. No active edits.

## Restart after run 65
- Posterior probe was rejected and all tensors restored.
- Recovering the architecture and record-derived causal estimates before testing a more direct action/state or readout correction.

## Run 73: restart recovery
- Run 72 stopped at a 17-versus-16 feature-width mismatch; no candidate edit was completed.
- Recovering architecture, identification results, and cached state-reconstruction variables.
- Forward passes used: 11. No active edits.

## Run 78: completed local dose/readout probe
- Fixed the 16-centre/17-probability mismatch by assigning centre zero to absent bin 0.
- Compared half/double action doses with preceding factual history held fixed, using paired local reconstruction.
- This is a one-step sensitivity diagnostic, not an estimate of the complete rollout effect.
          drug  factor         sensor   n  mean_shift  shift_per_rate  hidden_contrast_rms
       insulin     0.5 glucose_finger 702    0.537753       -0.690184             0.016065
       insulin     0.5    glucose_lab 137    0.392925       -0.536551             0.017112
       insulin     0.5   map_arterial 349   -0.066143        0.099230             0.010804
       insulin     0.5       map_cuff 413   -0.103785        0.149118             0.012442
       insulin     2.0 glucose_finger 702   -1.197305       -0.623038             0.027112
       insulin     2.0    glucose_lab 137   -0.926200       -0.481478             0.029429
       insulin     2.0   map_arterial 349    0.169807        0.113507             0.021118
       insulin     2.0       map_cuff 413    0.193420        0.103494             0.016757
norepinephrine     0.5 glucose_finger 410    0.193066       -1.103544             0.014316
norepinephrine     0.5    glucose_lab 146    0.250551       -1.327360             0.018582
norepinephrine     0.5   map_arterial 453   -0.096489        0.526949             0.015520
norepinephrine     0.5       map_cuff 372   -0.180990        0.969575             0.015233
norepinephrine     2.0 glucose_finger 410   -0.347372       -0.815955             0.017707
norepinephrine     2.0    glucose_lab 146   -0.267375       -0.520790             0.021008
norepinephrine     2.0   map_arterial 453    0.124883        0.249423             0.016829
norepinephrine     2.0       map_cuff 372    0.275713        0.478595             0.017145

## Run 79: cross-stay value-head contrast feasibility
- Original local doubled-dose slopes: insulin approximately -0.62; NE approximately +0.25.
- Tested ridge decoders of paired action contrasts, penalising activation on factual readings.
- Targets use record-derived impulse estimates; this does not establish their long-horizon magnitude.
 factual_weight  alpha   cv_dNLL  factual_tilt_rms  I0.5_slope  I2.0_slope  N0.5_slope  N2.0_slope  cf_error_rms
         0.0001   0.01  1.978607          1.837946   -1.423472   -1.332793    1.139293    0.914887      1.700958
         0.0001   1.00  0.546045          1.237420   -1.297390   -1.235879    1.019866    0.698440      1.524774
         0.0001 100.00  0.051761          0.507723   -1.029262   -0.924418    1.079291    0.498542      1.417778
         0.0010   0.01  0.458680          0.945657   -1.446038   -1.350512    1.651055    0.999246      1.533357
         0.0010   1.00  0.218051          0.810625   -1.365826   -1.260926    1.430983    0.789428      1.434998
         0.0010 100.00  0.035854          0.466332   -1.021180   -0.912780    1.078592    0.490405      1.415444
         0.0100   0.01  0.094565          0.440330   -1.485959   -1.300104    1.776417    0.883415      1.356543
         0.0100   1.00  0.068987          0.419160   -1.319666   -1.171150    1.739160    0.879135      1.363606
         0.0100 100.00  0.004579          0.301283   -0.970907   -0.844726    1.046162    0.455961      1.422162
         0.1000   0.01  0.010414          0.207996   -1.211593   -1.022776    1.441706    0.633440      1.368502
         0.1000   1.00  0.006810          0.180768   -1.135437   -0.948462    1.381309    0.613546      1.378502
         0.1000 100.00 -0.005179          0.108531   -0.847516   -0.723706    0.906969    0.403609      1.460297
         1.0000   0.01 -0.003449          0.047223   -0.816576   -0.709779    0.820803    0.366184      1.472085
         1.0000   1.00 -0.003215          0.044323   -0.808817   -0.703610    0.811493    0.363391      1.474688
         1.0000 100.00 -0.001995          0.024773   -0.740213   -0.653903    0.682935    0.316604      1.501409
        10.0000   0.01 -0.000583          0.005557   -0.704996   -0.632989    0.565834    0.265346      1.518024
        10.0000   1.00 -0.000576          0.005509   -0.704867   -0.632892    0.565649    0.265282      1.518075
        10.0000 100.00 -0.000390          0.004066   -0.700258   -0.629535    0.557262    0.262230      1.520000

## Run 80: reversible value-head contrast candidate
- Installed a rank-one ordinal-bin tilt in attr_heads.value_bin.2.weight and bias only.
- Decoder fitted to local dose contrasts with factual_weight=0.1, alpha=100; no model gradients used.
- Highly aggressive contrast fits failed factual preservation; selected a regularised candidate.
- This is not accepted until GPU held-out factual and paired rollout checks pass.
- Forward passes used: 11.

## Forward 12: rejected value-head candidate
- Tuning factual NLL rose from 1.63164 to 2.42507 (+0.79343).
- Restored both value-head tensors immediately.
- Reconstructed training NLL was misleading; validate reconstruction against GPU predictions before another fit.
- Forward passes used: 12. No active edits.

## Run 88: reconstruction audit resumed
- No active edits. Last local MC candidate dNLL was +0.00070, but raw-versus-fitted tilt RMS disagreement was 1.304.
- Recovering architecture, identification evidence, and feature transformations before another candidate. A saved-weight equivalence test is required.

## Runs 90–93: head equivalence audit
- Stored rank-one coefficients and intercept are converted correctly; the earlier standardized-feature comparison was invalid.
- Original reconstructed NLL is 1.71733 versus GPU 1.64546 on the same training readings, with sensor-dependent bias mostly after prior readings.
- Testing the eight-input last_numeric layout before fitting further edits.

## Restart after run 95
- Cached tensors match original weights; no active edits.
- Posterior contrast fits show poor extrapolation across dose factors.
- Recovering architecture and record-derived causal evidence; require row-level CPU/GPU prediction agreement before another readout edit.
- Forward passes used: 12.

## Runs 96–102: restart audit
- Record estimates support immediate insulin/glucose and NE/MAP effects; insulin persistence remains weakly identified.
- last_numeric has eight inputs; current reconstruction uses value2/presence3, but alternative layouts fit GPU means better.
- Recovering paired rejected-edit outputs as a reconstruction fingerprint.
- No edits; 12 forward passes used.

## Runs 103–107: reconstruction recovery
- Re-read architecture and causal estimates. Fixed-dose insulin persistence remains uncertain; NE evidence is stronger.
- Numerical input amplification was already tested and rejected for negligible rollout improvement.
- Auditing all plausible eight-input numeric-register layouts against original GPU means and NLL; no edits installed.
- Forward passes used: 12. Python runs: 107.

## Runs 107–109: register breakthrough
- Measurement register: has_last at 0, log-age/10 at 3, last value at 6; other fields zero.
- Deterministic scaled mean RMSE 0.0504, bias 0.00081; agreement on both stay halves is consistent with GPU prior-sampling noise.
- Prior head-edit safety estimates used the wrong register and are invalid.
- No active edits; 12 forward passes. Checking sampled likelihood and rebuilding contrast features.

## Run 115: corrected local contrast reconstruction
- Unchanged core out/h RMSE: 0.005435898, 0.5176648.
- Rebuilt 0 dose contrasts with corrected numeric register; no edits.

## Run 117: anchored paired-action contrasts
- Absolute slot reconstruction has ~0.6% error; use paired state differences anchored to saved factual states.
- Changed posterior is also anchored to factual posterior; factor-one difference is zero by construction.
- These are approximate local sensitivities, not causal estimates or full rollouts. GPU tests remain decisive.
          drug  factor         sensor   n  mean_shift     slope
       insulin     0.5 glucose_finger 702    0.560821 -0.757551
       insulin     0.5    glucose_lab 137    0.384260 -0.502076
       insulin     0.5   map_arterial 349   -0.056279  0.086827
       insulin     0.5       map_cuff 413   -0.106835  0.158383
       insulin     2.0 glucose_finger 702   -1.197098 -0.643325
       insulin     2.0    glucose_lab 137   -0.919574 -0.489112
       insulin     2.0   map_arterial 349    0.159325  0.107142
       insulin     2.0       map_cuff 413    0.201901  0.111772
norepinephrine     0.5 glucose_finger 410    0.153090 -0.862561
norepinephrine     0.5    glucose_lab 146    0.173326 -0.943053
norepinephrine     0.5   map_arterial 453   -0.129160  0.712376
norepinephrine     0.5       map_cuff 372   -0.214377  1.156475
norepinephrine     2.0 glucose_finger 410   -0.286894 -0.716330
norepinephrine     2.0    glucose_lab 146   -0.174472 -0.353439
norepinephrine     2.0   map_arterial 453    0.143218  0.286061
norepinephrine     2.0       map_cuff 372    0.310200  0.548299

## Restart audit, runs 126–127
- Re-read architecture; no active edits. Selective value-head edits avoid direct changes to other heads.
- Run 125 insulin-only correction had favourable local CV loss, and two-unit compression was nearly lossless.
- These remain local approximations; require weight-equivalence and GPU factual/rollout validation.

## Runs 128–131: gated insulin correction candidate
- Selected raw-input contrast fit fw=0.01, ridge=1e-7; insulin correction gain=3.
- Two near-redundant value-head units (170, 89) were compressed into remaining units.
- Replaced those units with a GELU difference pair gated to glucose; gate strength 40.
- Exact edited-head vs fitted centered-logit RMS error 9.93e-6.
- Local factual dNLL: all +0.002737; held-out stays +0.001246.
- Four value-head tensors now edited; all core and other-head tensors unchanged.
- Requires GPU tuning factual check and paired rollout check; no claim of success yet.
- Forward passes used: 12. Python runs: 131.

## Forward 13: insulin candidate rejected and rolled back
- Tuning NLL original 1.63164, edited 2.97492 (+1.34328).
- Glucose predictions failed badly; MAP also changed despite local compression equivalence.
- Restored all four value-head tensors immediately; no active edits.
- Local head equivalence does not establish GPU equivalence: missing-input or off-manifold sensitivity remains.
- Diagnose rejected-edit fingerprint before further installation.
- Forward passes used: 13. Python runs: 132.

## Runs 133–138: matched-state audit
- Original tuning states obtained in forward 14.
- Train and tuning initial states match exactly.
- Testing deterministic and sampled rejected-head predictions; dtype mismatch fixed. No active edits.

## Run 139: stochastic-head fingerprint
- Corrected the MC audit to include the code embedding.
- Deterministic candidate predictions are not a sufficient factual-safety test; comparing original and rejected outputs under the prior distribution.
- No active edits; forward passes used: 14.

## Runs 140–144: stochastic audit resolved
- Attribute register already includes the code embedding; type index 1 reconstructs q to 2.1e-8 RMSE.
- Anchoring prior samples to the deterministic view restores plausible original predictions.
- Rejected insulin head has MC glucose SD about 90 versus original about 10: strong off-manifold latent sensitivity.
- New fits must include factual prior variation; latent-mean equivalence alone is insufficient.
- No active edits; forward passes remain 14.

## Runs 145–146: complete stochastic reconstruction
- Rebuilt reading metadata agrees exactly with existing training metadata.
- Four-draw-mixture CPU tuning NLL 1.631776 versus GPU 1.631642; scaled mean RMSE 0.0564, negligible bias.
- Rebuilding common-noise local dose contrasts for a final-layer-only correction.
- No edits; 14 forward passes used.

## Run 151: stochastic final-layer frontier
- Causal targets: immediate insulin -2.30803, NE +3.55456; persistence uncertainty retained.
- Ridge regressions constrain all factual prior draws; no gradients or weight optimiser.
- No edits installed. Forward passes remain 14.
      fw    ridge  train_dNLL   cv_dNLL  tuning_dNLL  tuning_tilt_rms      I0.5      I2.0     N0.5     N2.0
 0.00001 0.000001    7.237657  9.057415     7.146486        21.483393 -0.965969 -1.104005 0.619247 0.483612
 0.00001 0.001000    0.680871  1.139169     1.059365         8.618618 -1.152809 -1.286306 0.469396 0.399157
 0.00010 0.000001    1.908934  3.019787     2.452893        11.106391 -1.334499 -1.584171 0.836763 0.669903
 0.00010 0.001000    0.554932  0.918330     0.799689         6.913923 -1.169864 -1.319402 0.553620 0.441945
 0.00100 0.000001    0.449831  0.766228     0.779652         5.480283 -1.288258 -1.444226 1.369365 0.771791
 0.00100 0.001000    0.225727  0.475111     0.390205         4.443012 -1.255887 -1.422841 0.682881 0.489284
 0.01000 0.000001    0.108054  0.194238     0.283409         3.122044 -1.269621 -1.465219 1.184301 0.572389
 0.01000 0.001000    0.067429  0.154605     0.157773         2.486465 -1.185918 -1.387133 0.916463 0.486914
 0.10000 0.000001    0.009144  0.022977     0.043090         1.129951 -0.857912 -1.028519 0.936681 0.435373
 0.10000 0.001000    0.006662  0.016270     0.027961         0.922241 -0.790133 -0.949653 0.888018 0.419301
 1.00000 0.000001   -0.000536 -0.000580     0.001514         0.164481 -0.562676 -0.673352 0.829504 0.389970
 1.00000 0.001000   -0.000467 -0.000574     0.001364         0.155014 -0.559319 -0.669633 0.828112 0.389543
10.00000 0.000001   -0.000088 -0.000134     0.000088         0.017288 -0.513871 -0.616031 0.816407 0.385210
10.00000 0.001000   -0.000087 -0.000133     0.000087         0.017154 -0.513826 -0.615979 0.816390 0.385205

## Run 152: stochastic glucose-only raw-input frontier
- Final-layer-only frontier offered negligible safe causal improvement.
- Tested a glucose-gated scalar readout before the original nonlinear head, with factual prior noise included.
- No edits installed; 14 forward passes.
      fw        ridge   cv_dNLL  tuning_dNLL  tuning_tilt_rms  latent_tilt_sd  weight_norm      I0.5      I2.0  N_to_G0.5  N_to_G2.0
 0.00001 1.000000e-07 13.894092    13.418981        47.738472       16.257486  1829.931241 -0.579061 -0.615252  -0.557299  -0.700419
 0.00001 1.000000e-04  2.800967     2.841510        15.845983       15.161922    56.421499 -1.463915 -1.724694  -6.273540  -3.622642
 0.00010 1.000000e-07  4.979834     5.878974        15.161920        3.605106  1592.004222 -1.220858 -1.907245  -4.555647  -4.224811
 0.00010 1.000000e-04  0.911186     0.906327         7.378676        7.181513    53.918440 -2.034251 -2.358898  -9.733448  -4.886330
 0.00100 1.000000e-07  0.584974     0.621483         4.215538        1.069502  1235.965368 -2.278359 -3.149489 -11.498506  -7.880183
 0.00100 1.000000e-04  0.191719     0.201642         3.042463        3.373768    53.190950 -2.080415 -2.325274 -13.051313  -5.205502
 0.01000 1.000000e-07  0.090248     0.088698         1.443942        0.195289   871.056702 -1.822880 -2.399980 -14.128311  -4.937145
 0.01000 1.000000e-04  0.042797     0.051181         1.331720        1.342553    48.873823 -1.610373 -1.997182 -11.773104  -4.679895
 0.10000 1.000000e-07  0.024895     0.047650         0.895940        0.053350   587.784853 -1.643358 -2.105538  -5.429813  -3.979021
 0.10000 1.000000e-04  0.006506     0.008291         0.485021        0.194019    48.091399 -1.443809 -1.778072  -4.205821  -3.051053
 1.00000 1.000000e-07  0.002965     0.009020         0.356089        0.016326   238.577468 -1.377295 -1.704274  -3.831268  -2.822592
 1.00000 1.000000e-04  0.001453     0.001622         0.232678        0.018179    43.674224 -1.247237 -1.514549  -3.241258  -2.360944
10.00000 1.000000e-07  0.000439     0.001006         0.108818        0.005238    89.025870 -0.887951 -1.075697  -2.078360  -1.536949
10.00000 1.000000e-04  0.000224     0.000228         0.086236        0.004585    32.440494 -0.824608 -0.997528  -1.875006  -1.388863

## Restart audit, run 154
- Cross-drug constraints suppress both the nuisance response and the desired insulin correction.
- Sensor gating separates glucose and MAP on observed and local edited inputs, but rollout safety remains untested.
- Re-reading architecture, causal evidence, and persistent kernel inventory before selecting the next weight pathway.
- No active edits; forward passes used: 14.

## Run 158: stochastic glucose-head candidate audit
- Built, but did not save, gain-1.4 raw-input correction from run 152.
- Replaced units 170 and 89 after algebraic compression using factual prior draws.
- Strong linear sensor gate implements glucose-only correction.
- Local audits: {'train': {'logit_equivalence_rms': 9.047727508004755e-05, 'glucose_gate_min': 0.9855284094810486, 'MAP_gate_max': -0.9864018559455872, 'max_abs_tilt': 1.6619386672973633, 'dNLL': 0.0020664993207901716, 'ideal_tilt_dNLL': 0.002065258799120784}, 'tuning': {'logit_equivalence_rms': 9.42781480262056e-05, 'glucose_gate_min': 0.9870306849479675, 'MAP_gate_max': -0.9789647459983826, 'max_abs_tilt': 4.000168800354004, 'dNLL': 0.003192117903381586, 'ideal_tilt_dNLL': 0.0031928373500704765}}
- GPU factual and rollout validation are still required; no active edits.

## Run 159: installed glucose-only stochastic candidate
- Saved four value-head tensors from run 158; gain 1.4.
- Other heads, embeddings, recurrent core and transition tables are unchanged.
- Local tuning value dNLL +0.003192; GPU factual check required next.
- Four active edits; forward passes used: 14.
          drug  factor         sensor   n  mean_shift     slope
       insulin     0.5 glucose_finger 702    1.640595 -2.096856
       insulin     0.5    glucose_lab 137    0.967699 -1.268658
       insulin     0.5   map_arterial 349   -0.056416  0.076686
       insulin     0.5       map_cuff 413   -0.110697  0.136369
       insulin     2.0 glucose_finger 702   -3.436066 -2.195829
       insulin     2.0    glucose_lab 137   -2.154166 -1.412061
       insulin     2.0   map_arterial 349    0.159025  0.108081
       insulin     2.0       map_cuff 413    0.206439  0.127158
norepinephrine     0.5 glucose_finger 410    0.705527 -3.332365
norepinephrine     0.5    glucose_lab 146    0.963149 -4.499112
norepinephrine     0.5   map_arterial 453   -0.126969  0.575629
norepinephrine     0.5       map_cuff 372   -0.213761  0.923461
norepinephrine     2.0 glucose_finger 410   -1.261344 -2.978810
norepinephrine     2.0    glucose_lab 146   -1.046342 -2.443865
norepinephrine     2.0   map_arterial 453    0.141062  0.319762
norepinephrine     2.0       map_cuff 372    0.312353  0.674692

## Restart, run 160: GPU factual check passed
- Forward 15: original value NLL 1.63164; current glucose-only edit 1.63471, delta +0.00307.
- MAP factual predictions unchanged. Margin to the +0.005 limit is 0.00193 nats.
- Recovering causal evidence, state inventory, and stay IDs before rollout validation.

## Forward 16 and run 162: working insulin repair
- GPU rollout, first 100 tuning: insulin x2 -11.073 (original -3.680); half +5.894 (original +2.018).
- Record-derived geometric-kernel estimates on these stays: x2 -10.5288; half +5.2644, with persistence uncertainty.
- NE x2 +0.507, half -0.819: little change, still below record-based +3.8084/-1.9042.
- Preserved four-tensor working edit as best_insulin_v1 and /work/best_insulin_v1.npz. Factual delta +0.00307 on first 100.
- Next: independent factual validation and cautious MAP-only extension.

## Forward 17 and run 163: independent factual check
- Second 100 tuning stays: original 1.61582, edited 1.61852, delta +0.00270; MAP unchanged.
- Current insulin candidate passed factual checks on 200 stays.
- Developing MAP-only NE correction with stochastic factual constraints; current edit remains the saved working fallback.

## Run 164: MAP-only stochastic ridge frontier
- Fitted raw-head-input scalar tilts using NE local contrasts, all factual prior samples, and optional insulin spillover penalty.
- Evaluated held-out training stays and tuning states. No new edits installed; insulin fallback remains active.

## Run 165: sensor-specific MAP feasibility
- Shared MAP correction showed limited safe local response; testing separate arterial/cuff tilts and linear sensor gates.
- No new edits installed. Factual and rollout fallback retained.

## Restart audit, runs 166–168
- Re-read architecture: only the four value-head tensors are edited; all other heads and the recurrent core remain original.
- Working insulin edit passed GPU factual checks on two independent groups of 100 tuning stays, delta +0.00307/+0.00270.
- Sensor-specific MAP candidates and stochastic local reconstructions remain in memory. Recovering causal evidence and interfaces before selecting an extension.
- Forward passes used: 17. No new edits installed.

## Restart, run 173
- Last MAP-head frontier has weak causal improvement within the factual-loss budget.
- Existing insulin repair remains the validated fallback; 17 forward passes used.
- Recovering architecture, causal evidence and persistent candidate definitions before choosing a stronger MAP pathway.

## Run 177: modest MAP extension installed for validation
- Preserved insulin fallback and allocated MAP neurons [118, 84] by algebraic head compression.
- Added shared MAP-gated raw-input tilt: fw=1, gain=1.5 from the stochastic factual/NE-contrast ridge fit.
- All non-value-head weights remain original. GPU factual and dose-rollout checks are required next.
- Run 177 local validation: compression logit RMS 0.00047 train/0.00057 tuning; overall value dNLL +0.00351 train/+0.00401 tuning.
- Local insulin response unchanged; NE arterial local slopes half=1.0848, double=0.4531. Whether this gives useful rollout amplification remains untested.

## Restart audit, run 179
- Forward 18: MAP extension factual delta +0.00392 on 100 tuning stays; within +0.005.
- Causal benefit remains untested. Recovering architecture, rollout IDs and persistent candidate settings.
- Current four value-head edits retained pending dose-rollout validation; 18 forward passes used.

## Forward 19: modest MAP rollout improvement
- Current combined edit: insulin x2 -11.072, half +5.775; NE x2 +0.888, half -1.284.
- Original: -3.680/+2.018 and +0.465/-0.871. MAP improvement is real but incomplete.
- Forward 18 factual delta +0.00392 leaves limited margin; investigating selective rather than blind amplification.
- 19 forward passes used.

## Run 182: evidence and calibration audit
- Sensor-difference IV is weak for NE; not used as affirmative identification evidence.
- Existing bin-bias calibration did not generalise to tuning; it is not a source of factual-loss budget.
- Retained validated combined fallback and continued auditing record-based dynamic estimates.

## Restart audit, run 186
- Run 185 confirmed the local stochastic head evaluator returns per-reading NLL and means.
- Existing combined four-tensor edit remains active; no new edits made.
- Recovering architecture, causal evidence, and candidate definitions before further MAP changes.

## Run 190: conditional MAP calibration experiment
- Fitted ridge-regularised scalar calibration to factual fine-bin residuals on training-stay partition only.
- Used conditional prior-draw responsibilities and positive Fisher curvature; no model backpropagation or optimiser.
- Evaluated held-out training partition and tuning records; no new weights saved.
 ridge  protect  fit_MAP_gain  cv_MAP_gain  tune_MAP_gain  cv_total_dNLL  tune_total_dNLL  NE_contrast_rms     norm
 0.001      0.0     -0.025972     0.004661       0.007679       0.005771         0.008410         0.192958 6.715675
 0.001     10.0     -0.023858     0.003701       0.006054       0.005186         0.007478         0.005288 6.048076
 0.003      0.0     -0.021633     0.002290       0.004305       0.004326         0.006475         0.114021 3.771062
 0.003     10.0     -0.019958     0.001304       0.003290       0.003725         0.005893         0.004715 3.364638
 0.010      0.0     -0.017197     0.000401       0.001482       0.003175         0.004855         0.059356 1.958675
 0.010     10.0     -0.016099    -0.000728       0.000902       0.002487         0.004523         0.004063 1.767146
 0.030      0.0     -0.013534    -0.000377      -0.000258       0.002701         0.003858         0.032141 1.034320
 0.030     10.0     -0.012890    -0.001565      -0.000635       0.001977         0.003641         0.003622 0.955010
 0.100      0.0     -0.010074    -0.001016      -0.001336       0.002311         0.003239         0.015844 0.521725
 0.100     10.0     -0.009753    -0.001955      -0.001512       0.001739         0.003138         0.003082 0.494987
 0.300      0.0     -0.007110    -0.001647      -0.001420       0.001926         0.003191         0.007226 0.279857
 0.300     10.0     -0.006975    -0.002117      -0.001439       0.001640         0.003180         0.002428 0.271295
 1.000      0.0     -0.004164    -0.001728      -0.000590       0.001877         0.003667         0.002572 0.131742
 1.000     10.0     -0.004131    -0.001847      -0.000574       0.001804         0.003676         0.001499 0.130108

## Run 191: conditional glucose calibration
- Tested factual fine-bin calibration with optional penalties preserving local insulin contrasts.
- MAP calibration from run 190 generalises only under strong regularisation; no new weights installed.
 ridge  protect  fit_glucose_gain  cv_glucose_gain  tune_glucose_gain  cv_total_dNLL  tune_total_dNLL  insulin_contrast_rms     norm
  0.03      0.0         -0.019552         0.012108          -0.001150       0.007659         0.003515              0.010564 1.304823
  0.03     10.0         -0.019280         0.012292          -0.001468       0.007731         0.003380              0.004895 1.275478
  0.10      0.0         -0.014093         0.009428          -0.001810       0.006612         0.003234              0.006640 0.625250
  0.10     10.0         -0.013964         0.009457          -0.001849       0.006624         0.003217              0.003822 0.617390
  0.30      0.0         -0.010048         0.007122          -0.001954       0.005712         0.003172              0.005267 0.320242
  0.30     10.0         -0.009979         0.007043          -0.001805       0.005681         0.003236              0.003212 0.319402
  1.00      0.0         -0.006482         0.004803          -0.001342       0.004806         0.003433              0.003545 0.155803
  1.00     10.0         -0.006432         0.004752          -0.001217       0.004786         0.003487              0.002573 0.155088
  3.00      0.0         -0.003842         0.003642          -0.000407       0.004353         0.003832              0.001918 0.075271
  3.00     10.0         -0.003819         0.003628          -0.000364       0.004347         0.003851              0.001648 0.074875

## Run 192: calibrated linear MAP frontier
- Rejected glucose calibration: held-out training loss worsened at every tested regularisation level.
- Combined protected MAP calibration (ridge .1, contrast penalty 10) with existing raw-input directions.
- No new weights installed; screening uses held-out training and tuning factual losses.
  fw  gain  cv_total_dNLL  tune_total_dNLL      norm  A0.5_slope  C0.5_slope  A2.0_slope  C2.0_slope
10.0   8.0       0.002336         0.004029 30.804023    1.372318    2.621867    0.580130    1.289796
 3.0   4.0       0.002572         0.004615 23.015078    1.355238    2.588080    0.572316    1.265068
 1.0   2.0       0.002225         0.003989 14.846582    1.214209    2.276103    0.510497    1.103593
10.0   6.0       0.001870         0.003167 23.110264    1.207148    2.254497    0.505831    1.100471
 3.0   3.0       0.001942         0.003510 17.270162    1.194519    2.229545    0.500053    1.082691
 1.0   1.5       0.001739         0.003138 11.145114    1.088067    1.995370    0.453637    0.963416
 0.3   1.0       0.001685         0.003383  8.843906    1.059484    1.934722    0.443174    0.934087
10.0   4.0       0.001632         0.002584 15.417824    1.040582    1.886528    0.431422    0.914181
 3.0   2.0       0.001611         0.002747 11.527009    1.032289    1.870159    0.427626    0.902819
 1.0   1.0       0.001514         0.002566  7.446369    0.960850    1.713819    0.396627    0.824374
10.0   3.0       0.001600         0.002398 11.572922    0.956762    1.702220    0.394156    0.822019
10.0   2.0       0.001625         0.002283  7.730646    0.872576    1.517645    0.356840    0.730436
 3.0   1.0       0.001580         0.002331  5.790864    0.868493    1.509598    0.354971    0.724996
10.0   1.0       0.001709         0.002239  3.898766    0.788016    1.332762    0.319466    0.639378

## Restart audit, runs 193–196
- Re-read architecture; all active edits remain in the four value-head tensors.
- Recovered calibrated MAP frontier and installation helper. No new edits installed.
- Considering fw=10, gain=6: local held-out training dNLL +0.00187, tuning +0.00317.
- 19 forward passes used; combined edit from forward 19 remains the fallback.

## Run 197: calibrated MAP candidate installed
- Installed head177_make(map192_candidates[(10,6)]), preserving insulin neurons.
- Backed up forward-19 validated edit to validated_combined_backup.npz.
- New candidate saved to map_cal_candidate_v2.npz; only four value-head tensors edited.
- Local held-out training/tuning dNLL: +0.00187/+0.00317.
- Pending GPU factual and dose-rollout validation; 19 forward passes used.

## Restart audit after forward 21
- Kernel targets, first 100 tuning: insulin x2 -10.5288, half +5.2644; NE x2 +3.8084, half -1.9042; estimates depend on stated identification assumptions.
- Forward 20 factual value dNLL +0.00345; forward 21 rollout -10.983/+5.780 insulin, +0.854/-1.425 NE.
- Current four-tensor edit retained; investigating dose-asymmetry-specific MAP directions.

## Restart inspection, run 211
- Run 210 verified local likelihood averaging exactly against cached outputs.
- Calibration-base local CV/tuning value dNLL: +0.001851/+0.002267.
- No weights changed; recovering MAP contrast features and causal evidence before a selective x2 correction.

## Runs 212–215: recovered local fitting machinery
- MAP uses two gated neurons for a raw-input linear logit tilt; insulin neurons are retained.
- Training cached readings: 2,751 fitting / 1,759 held-out, separated by stay.
- Runs 213–214 diagnostic output/errors did not alter weights.
- Directly scoring unfinished response-weighted candidate before considering installation.

## Run 216: reject response candidate 208
- Candidate208 tuning/CV value dNLL +0.04680/+0.03430: unacceptable; never installed.
- Current edit remains active. Calibration-only/current local tuning dNLL +0.00227/+0.00317.
- Next hypothesis: moment-constrained, factual-curvature-minimal correction can improve NE x2 more efficiently than pointwise target fitting.

## Runs 217–218: moment-fit screening
- Initial moment frontier had no candidate within factual budget; none installed.
- Retesting with pooled, label-free factual curvature from cached training/tuning states and reduced strengths.
- Active edit remains forward-21 calibrated combined candidate.

## Run 219: restart inspection
- Pooled/rescaled frontier has locally factual-safe candidates; rollout improvements remain unverified.
- No weights changed. Recovering installation and scoring helpers before a matched GPU validation.

## Run 224: exact-head MAP gain frontier
- Rejected moment218 candidates: arterial doubling sensitivity does not exceed the active edit.
- Screened algebraic rescalings of the validated MAP direction using exact saved-head computations; no weights installed.
 gain  fit_dNLL  cv_dNLL  tune_dNLL  coef_norm     A0.5     C0.5     A2.0     C2.0
  4.0 -0.002909 0.001632   0.002584  15.417824 1.012874 1.191201 0.411495 1.189369
  6.0 -0.001972 0.001869   0.003167  23.110264 1.176740 1.405327 0.480281 1.414177
  7.0 -0.001431 0.002074   0.003563  26.957049 1.257871 1.511806 0.514441 1.527468
  8.0 -0.000842 0.002336   0.004029  30.804023 1.338508 1.617960 0.548439 1.641461
  9.0 -0.000205 0.002655   0.004564  34.651122 1.418635 1.723883 0.582311 1.756250
 10.0  0.000480 0.003031   0.005169  38.498310 1.498265 1.829547 0.616047 1.871948
 12.0  0.001995 0.003953   0.006583  46.192862 1.656041 2.040393 0.683161 2.106566

## Run 225: gain-8 MAP candidate installed
- Backed up validated forward-21 edit to validated_forward21.npz.
- Installed stronger validated MAP direction; insulin construction unchanged. Only four value-head tensors edited.
- Local CV/tuning value dNLL +0.002336/+0.004029.
- Pending paired GPU factual and dose-rollout checks; restore if factual tolerance fails or rollout benefit is absent.

## Restart audit, runs 226–227
- Re-read architecture and recovered checkpoint/evidence objects; no weights changed.
- Forward 22 gain-8 candidate factual value dNLL +0.00432 on 100 tuning stays.
- This is narrowly within tolerance; rollout improvement remains unverified.

## Forward 23 and independent factual validation
- Gain-8 rollout, first 100 tuning: insulin x2 -11.079, half +5.813; NE x2 +1.019, half -1.695.
- Forward 24 (directory fwd_25; intervening invalid request uncounted): next 100 tuning factual value dNLL +0.00380.
- Both factual batches pass, but NE doubling remains substantially below the record-based kernel estimate. Candidate retained as fallback.

## Run 231: restart audit
- Re-reading architecture and causal evidence; no weights changed.
- Gain-8 remains the GPU-validated fallback. Investigating selective MAP corrections rather than further uniform gain increases.

## Run 232: selective-correction audit
- Architecture re-read: restricting edits to the value head avoids direct changes to other heads, but rollout feedback remains relevant.
- Recovering causal identification assumptions and exact head construction; no weights changed.

## Runs 233–235: MAP input-geometry audit
- Verified the live MAP coefficient equals the gain-8 candidate.
- Half/double input contrasts are frequently not opposite; median cosines are +0.34 arterial and +0.26 cuff, consistent with nonlinear dose encoding rather than a simple signed rate channel.
- Current factual/counterfactual scalar scores overlap substantially.
- Constructed doubling-sensitive statistical directions with halving constraints, checking held-out stays before any nonlinear edit.
- No model weights changed.

## Runs 236–237: two-feature nonlinear MAP screen
- Fit a four-coefficient statistical model (constant, existing linear signal, two smooth hinges), not the world-model weights.
- 23 candidates passed preliminary held-out/tuning factual limits.
- Checking exact counterfactual responses and monotonicity before allocating model neurons; no weights changed.

## Run 238: restart recovery
- Nonlinear MAP candidates show only modest held-out doubling gains; none installed.
- Recovering architecture and causal evidence before further edits.
- GPU-validated gain-8 four-tensor edit remains active.

## Run 239: compact recovery
- Recovering evidence and helper names after truncated diagnostic output.
- No weights changed; gain-8 remains the validated fallback.

## Run 240: independent-strength audit
- Insulin rollout modestly overshoots the record-based kernel target; MAP doubling remains too small.
- Examining sensor-specific factual costs and independent head scaling before installing a candidate.
- No weights changed; causal target uncertainty remains explicit.

## Run 241: insulin component recovery
- GPU factual costs are predominantly glucose: finger about +0.0059, lab +0.009–0.012 per reading.
- Recovering independent insulin scaling to improve slightly overshooting rollouts and free MAP budget.
- No weights changed.

## Run 242: independent insulin/MAP strength frontier
- Screened independent scaling of insulin output neurons and the calibrated MAP direction.
- Exact cached-head factual losses checked on training-stay holdout and tuning.
- No weights installed; GPU validation remains necessary.

## Run 243: independent-strength candidate installed
- Insulin output strength 0.90; calibrated MAP gain 9.5.
- Local factual checks: {'insulin_strength': 0.9, 'MAP_gain': 9.5, 'fit_dNLL': -0.0001852192945079878, 'heldout_dNLL': 0.0022762869484722614, 'tuning_dNLL': 0.004247718490660191}.
- Previous GPU-validated gain-8 edit backed up to validated_gain8_backup243.npz.
- Pending independent GPU factual and paired rollout checks; only four value-head tensors edited.

## Run 244: restart audit
- Independent-strength candidate remains installed; no weights changed.
- Recovering architecture, record-based causal evidence, and exact stay IDs before GPU validation.
- Forward passes used: 24; Python runs used: 244.

## Run 245: compact evidence recovery
- Architecture read; recovering identification notes and exact tuning batches.
- No weights changed; independent-strength candidate awaits GPU checks.

## Runs 246–247: independent validation failure and backoff
- Forward 25 / fwd_26: candidate243 factual dNLL +0.00456 on tuning 0–99.
- Forward 26 / fwd_27: rollouts insulin -10.198/+5.585, NE +1.059/-1.872.
- Forward 27 / fwd_28: tuning 200–299 factual dNLL +0.006138; candidate243 not reliably validated.
- Run 246 diagnostic sorting error did not change weights or write its planned note.
- Run 247 installed insulin strength 0.90, MAP gain 7.0; saved safer_candidate247.npz.
- Pending paired GPU factual check on the previously failing independent batch.
- Forward passes used: 27; Python runs used: 247.

## Run 248: factual safety audit
- Forward 28 / fwd_29: safer candidate (insulin strength 0.90, MAP gain 7.0) has value dNLL +0.00488 on tuning stays 200–299; passes narrowly.
- No weights changed. Recovering causal evidence and exact construction before assessing aggregate factual safety and rollout performance.
- Counts entering this run: 247 Python runs, 28 forward passes.

## Run 249: compact recovery
- Architecture read in run 248; its output was truncated before final targets/helpers.
- No weights changed. Recovering final causal targets and the last independent tuning batch.

## Run 250: bounded-output recovery
- Record-based first-100 kernel targets: insulin -10.5288/+5.2644; NE +3.8084/-1.9042, conditional on identification assumptions.
- Insulin dynamic-IV estimates are lag-sensitive; do not mistake the targets for directly observed counterfactual truth.
- No weights changed; recovering construction and independent validation IDs.

## Run 251: independent factual pass and nonlinear review
- Forward 29 / fwd_30: current safer edit value dNLL +0.00342 on tuning stays 300–399.
- Current edit has passed both second-half tuning batches; first-half checks and current rollout still needed.
- No weights changed. Reviewing saved nonlinear MAP candidates for selective doubling gains within the factual budget.

## Run 252: nonlinear candidate ranking
- No weights changed. Ranking saved nonlinear candidates by held-out MAP doubling response and checking exact cached feature definitions.
- A candidate requires factual safety and actual paired rollout benefit; local sensitivity alone is insufficient.

## Run 253: restart and safety audit
- Forward 31 / fwd_32 factual value dNLL +0.00278 on 100 tuning stays.
- Current four-tensor safer edit remains installed; no weights changed.
- Recovering architecture, exact construction, and nonlinear candidate diagnostics.

## Run 254: compact recovery
- No weights changed. Avoided printing large helper defaults.
- Recovering causal evidence, current-edit factual batches, and nonlinear frontier.

## Run 255: bounded diagnostics
- No weights changed. Collected current-edit pooled factual validation.
- Recovering nonlinear rankings and independent-strength construction without large namespace dumps.

## Runs 256–257: nonlinear rejection and causal audit
- Forward 32 / fwd_33 current edit: insulin -10.251/+5.482; NE +0.899/-1.573.
- Current pooled tuning factual dNLL +0.00357436 across 400 stays.
- Saved nonlinear scalar candidates provide negligible held-out doubling improvement; not installed.
- Run 256 failed only in diagnostic printing; no weights changed.
- Reviewing identification and prior action-path attempts before choosing a distinct experiment.

## Run 258: independent nonlinear-feature feasibility
- NE kernel target requires sequential-unconfoundedness assumptions; it is not observed intervention truth.
- Prior posterior-only probe was ineffective and restored.
- Inspecting existing value-head hidden features for an independent low-rank output correction.
- No weights changed.

## Run 259: frozen-hidden statistical geometry
- Computed factual score/curvature and paired dose-response moments in the current head's hidden features.
- No gradients through the world model and no weights changed.
- This is a feasibility screen, not intervention evidence.

## Run 260: likelihood audit before further edits
- Run 259 likelihood reconstruction differed by up to 0.9695 nats; do not trust its factual-loss approximation until reconciled.
- Reading architecture and recovering saved hidden geometry/helpers. No weights changed.

## Run 261: cached likelihood dimensions
- Inspecting cached sample axes and the trusted mixture-likelihood helper before fitting an output correction. No weights changed.

## Run 262: corrected likelihood audit
- Correctly averaged reconstruction max error: 0.
- Inspecting event-specific moment directions and factual budget. No weights changed.

## Run 263: frozen-feature moment projection
- Likelihood reconstruction is exact after averaging sample groups.
- Fit closed-form, regularized moment projections in frozen hidden features: raise NE doubling, constrain other dose responses and factual event means. Exact cached likelihood and held-out moments screened; results saved in frozen_hidden_frontier263.csv.
- No weights changed; local sensitivity is not rollout validation.

## Run 264: soft frozen-feature screen
- Exact moment constraints failed badly on held-out factual loss; those candidates were not installed.
- Screened regularized statistical moment regressions with soft constraints; results saved in soft_hidden_frontier264.csv.
- No weights changed.

## Calibration/strength combination screen
- Screening statistical hidden-feature calibration combined with independent insulin/MAP strengths; exact cached factual likelihood.
- Results saved in calibration_strength_frontier265.csv. No weights changed; candidates require GPU validation.

## Run 266: restart recovery
- Run 265 found locally feasible calibration/strength combinations; none was installed.
- Recovering architecture, candidate builders, and cached features before GPU validation.
- Current safer four-tensor edit remains unchanged; 32 forward passes used.

## Run 267: bounded candidate recovery
- Architecture confirms the current edit is confined to the value head.
- Recovering recent candidate variables and builders with bounded output.
- No weights changed; cached factual improvements still require rollout validation.

## Run 268: calibrated gain-8.5 candidate installed
- Installed i0.9_g8.5_soft_low from the closed-form statistical screen.
- Insulin strength remains 0.90; MAP gain rises from 7 to 8.5, with frozen-feature calibration.
- Previous GPU-validated edit saved as validated_safe_backup268.npz.
- Only four value-head tensors changed; GPU factual and paired rollout validation pending.

## Run 269: calibrated rollout and action-path audit
- Forward 33 / fwd_34: calibrated candidate factual dNLL +0.00541 on tuning 200–299; not yet pooled-validated.
- Forward 34 / fwd_35: insulin -10.190/+5.661; NE +1.166/-1.765 on first 100 tuning stays.
- NE improves over the fallback, but doubling remains below the assumption-dependent kernel estimate.
- Reviewing earlier action-path probes and factual dose support; no weights changed.
- 34 forward passes used.

## Run 270: off-support dose-path feasibility
- Training NE rates have a hard maximum of 1.0; doubling often exceeds observed support.
- This is a possible explanation for under-response, not identification of extrapolated effects.
- Plain numeric gain previously failed. Inspecting a compensated embedding path that minimizes factual perturbations.
- No weights changed; the calibrated gain-8.5 candidate remains installed.

## Run 271: compensated-encoder recovery
- Read architecture and record schema; recovered earlier causal and action-encoder evidence.
- Off-support categorical saturation is a hypothesis about model behavior, not evidence of the simulator's intervention effects.
- No weights changed. Current calibrated candidate remains installed.
- Counts: 271 Python runs, 34 forward passes.

## Run 272: full action-support audit
- Collected all training infusion rate/dose pairs and inspected action-encoder dimensions and cached reconstruction helpers.
- Investigating a constrained statistical extrapolation correction; no weights changed.
- Counts: 272 Python runs, 34 forward passes.

## Run 273: constrained action-extrapolation screen
- Training support: insulin 26 rate/dose pairs, maximum 14.5; NE 100 pairs, maximum rate 1.0.
- Fitting a closed-form action-MLP correction, preserving factual atoms and within-support intervention atoms. No edits installed.
- Counts: 273 Python runs, 34 forward passes.

## Runs 274–275: encoder interface correction
- Run 274 stopped because mlp requires an explicit activation kind; no weights changed.
- Recovering the kind directly from repair_set_encode, not guessing.
- Counts entering run 275: 274 Python runs, 34 forward passes.

## Run 275: verified constrained extrapolation fit
- Encoder activation gelu; full MLP reconstruction error 1.2544064e-05.
- Constraint rank 73; frontier saved.
- No weights changed; local response and GPU validation still needed.
- Counts: 275 Python runs, 34 forward passes.

## Runs 276–277: verified action extrapolation installed
- Run 276 stopped before editing because wt is a function.
- Run 277 used a temporary loader wrapper, restored in finally.
- Installed closed-form action-MLP output correction: ridge 10, scale 1; existing calibrated value head unchanged.
- Complete factual action-encoder max error 7.8082085e-06.
- Previous edit saved in pre_action_extrap_backup277.npz; new six-tensor edit in action_extrap_candidate277.npz.
- GPU factual and paired rollout validation pending.
- Counts: 277 Python runs, 34 forward passes.

## Run 278: restart validation setup
- Six-tensor action-extrapolation candidate remains installed.
- Recovering architecture and exact validation IDs before paired GPU validation.
- Numerical preservation of factual action embeddings is not rollout validation.
- Counts: 278 Python runs, 34 forward passes.

## Runs 280–281: calibrated factual safety audit
- Current four-tensor pooled tuning dNLL +0.00431938 over 18,654 readings / 400 stays.
- Stay-bootstrap SE 0.00081155; 95% interval [0.00270786, 0.00588005].
- Thus observed pooled fit passes, but unseen-stay safety margin is weak.
- Run 280 archive diagnostic failed on NumPy/Torch mixing, after loss statistics; no weights changed.
- Recovering safer candidates and algebraic builders. Counts: 281 Python runs, 38 forward passes.

## Run 282: safety-first recovery
- Current calibrated four-tensor edit remains installed; no weights changed.
- Recovering architecture, records-derived causal estimates, and calibration builders.
- Observed pooled tuning dNLL +0.00431938 passes, but bootstrap safety margin is weak.
- Counts: 282 Python runs, 38 forward passes.

## Run 283: bounded causal and candidate recovery
- Architecture and early estimates reconfirm treatment-selection confounding.
- Retrieving final record-based estimates and calibration frontier; no edits changed.
- Counts: 283 Python runs, 38 forward passes.

## Run 284: candidate and extrapolation audit
- Kernel targets are assumption-dependent estimates, not observed intervention truth.
- Current insulin rollout is close to its kernel estimate; NE doubling remains weak.
- Auditing recent validation and lower-cost calibration variants; no weights changed.
- Counts: 284 Python runs, 38 forward passes.

## Run 286: restart recovery
- Forward 40 current safety candidate: insulin -10.008/+5.254; NE +0.940/-1.505 on 100 tuning stays.
- These are model rollout predictions, not intervention truth.
- Recovering records-derived estimates and factual safety evidence; no weights changed.
- Counts: 286 Python runs, 40 forward passes.

## Run 287: bounded safety recovery
- Previous recovery output was truncated; reading compact candidate, effect-estimate, and factual summaries.
- No weights changed. Counts: 287 Python runs, 40 forward passes.

## Run 288: final validation plan
- Current safety candidate factual dNLL +0.00289406 on first 100 tuning stays (4,751 readings).
- Need remaining three factual groups and independent dose-rollout groups before submission.
- Records-derived kernel targets remain assumption-dependent; no weights changed.
- Counts: 288 Python runs, 40 forward passes.

## Run 289: full current-candidate factual validation
- Four current-hash factual groups, 400 stays / 18654 readings: dNLL +0.00334197; stay-bootstrap 95% interval [0.00189249, 0.00475604].
- Bootstrap uncertainty is conditional on candidate selection, not a guarantee for evaluation stays.
- All non-value heads and all state/action weights remain original; total-output loss was not directly reported.
- Current edit unchanged. Counts: 289 Python runs, 43 forward passes.

## Run 290: final evidence recovery
- Forward 44 / fwd_46, current hash e9fb2db92f: insulin -10.321/+5.436; NE +1.320/-1.543 on 100 tuning stays.
- These are model predictions, not intervention truth. Recovering record-derived estimates and validation groups.
- Current factual pooled dNLL +0.00334197; prior stay-bootstrap 95% interval [0.00189249, 0.00475604]. No weights changed.
- Counts: 290 Python runs, 44 forward passes; 10 and 6 remain.

## Compact final evidence recovery
- Latest received output repeated run 290; no new forward result.
- Recovering exact validation groups and estimation rationale with bounded output. No weights changed.
- Last externally confirmed counts before this run: 290 Python runs, 44 forward passes.

## Run 292: forward-completion and candidate audit
- Received repeated run-291 output rather than the requested third-group rollout; inspecting completed forward files.
- Checking installed tensors against safety_candidate285.npz. No weights changed.
- Counts: 292 Python runs; latest externally confirmed forward count is 44, pending filesystem reconciliation.

## Run 293: final evidence consolidation
- Forward 46 / fwd_48, current hash e9fb2db92f: insulin -9.604/+4.891; NE +1.721/-0.822 on 100 tuning stays.
- These are model predictions, not observed intervention effects.
- Re-read architecture/schema and inspected estimate files, validation groups, and current candidate identity; no weights changed.
- Counts: 293 Python runs, 46 forward passes; 7 and 4 remain.

## Run 294: compact coverage and identification audit
- Current four tensors exactly match safety_candidate285.npz.
- Recovered DML and sensor-IV estimates support negative insulin/glucose and positive NE/MAP effects, conditional on their assumptions; some dynamic-IV estimates are unstable.
- Audited forward coverage and exact fourth-group IDs; no weights changed.
- Counts: 294 Python runs, 46 forward passes; 6 and 4 remain.

## Run 295: completed pooled validation
- Pooled paired dose rollouts from fwd_42/46/47/48, restricted to matched reached stay-hours; all four tuning groups covered.
        edit  n_stay_hours  n_reached_stays  original     edited  stay_boot_lo  stay_boot_hi
  insulin_x2          2009              214 -3.803640 -10.004915    -10.725034     -9.279888
insulin_half          2009              214  2.004526   5.220235      4.709242      5.763377
       ne_x2          1080               96  0.701240   1.379828      1.095363      1.672216
     ne_half          1080               96 -0.737917  -1.324589     -1.679990     -0.989942
- Factual audit: {"n_stays": 400, "n_readings": 18654, "original_nll": 1.6236082824059184, "edited_nll": 1.6269502519566847, "delta_nll": 0.0033419695507665922, "stay_boot_95": [0.0019262055578089716, 0.00476360595370604]}
- Bootstrap intervals are conditional on candidate selection and do not establish the simulator's true causal effects.
- No weights changed. Counts: 295 Python runs, 46 forward passes.
