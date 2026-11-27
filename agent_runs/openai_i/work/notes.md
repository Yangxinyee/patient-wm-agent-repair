
# Weight-repair investigation
## Initial inspection — Python 1; forward 0
- No weights changed and no drug effects assumed.
- Inspect architecture, vocabulary, and record schema first.
- Establish effects from training records, accounting for dose selection.
- Validate any edit against paired rollouts and the factual NLL limit (original + 0.005 nats per reading).

## Inspection — Python 2; forward 0
- Architecture read: separate action encoder; recurrent action pathways; shared attribute embeddings and same-code last-value registers.
- Initial output truncated before vocabulary and sample rows; reprinted compact metadata and loaded records.
- Reserved fixed tuning subsets for development, independent checking, and final validation. No weights changed.

## Hourly analysis — Python 3; forward 0
- Measurements are at integer hours; infusions begin at hour + 0.0833.
- Built chronological panels with zero rate when no infusion is recorded.
- Preliminary associations below are diagnostic, not established effects; linear adjustment and dynamic fixed effects can be biased.
        sensor           drug            method      n  beta_per_rate  cluster_se
glucose_finger        insulin             naive 263668      17.929280    0.115173
glucose_finger        insulin  history-adjusted 263668      -1.889789    0.042105
glucose_finger        insulin history + stay FE 263668      -1.939217    0.041457
   glucose_lab        insulin             naive  52643      17.885480    0.164865
   glucose_lab        insulin  history-adjusted  52643      -1.973027    0.068524
   glucose_lab        insulin history + stay FE  52643      -2.059826    0.070363
  map_arterial norepinephrine             naive 152923     -14.511075    0.241974
  map_arterial norepinephrine  history-adjusted 152923       2.094215    0.085729
  map_arterial norepinephrine history + stay FE 152923       2.621544    0.085211
      map_cuff norepinephrine             naive 111949     -15.371100    0.234747
      map_cuff norepinephrine  history-adjusted 111949      -0.673979    0.181196
      map_cuff norepinephrine history + stay FE 111949       1.578414    0.174600

## Nonlinear identification — Python 4; forward 2
- Original development factual NLL 1.66206; limit 1.66706.
- Original rollouts (99 stays): insulin x2 -1.864, half +0.975; NE x2 +0.342, half -0.032.
- Crossfit dose and outcome nuisance models by stay, using only pre-dose history. Active-dose estimates still require exchangeability and attention to observation selection and nuisance-model error.
        sensor           subset      n      beta  cluster_se
glucose_finger       all active 263668 -1.986075    0.044771
glucose_finger           hour 0   9307 -2.004139    0.243609
glucose_finger        hours 1-5  48376 -1.811866    0.104447
glucose_finger       hours 6-11  63523 -1.945429    0.091394
glucose_finger      hours 12-23 142462 -2.052531    0.059515
glucose_finger both sensors now  51056 -1.872719    0.095800
   glucose_lab       all active  52643 -2.101326    0.073338
   glucose_lab           hour 0   1868 -1.747917    0.457723
   glucose_lab        hours 1-5   9655 -1.874173    0.178218
   glucose_lab       hours 6-11  12707 -2.138924    0.153661
   glucose_lab      hours 12-23  28413 -2.175754    0.097108
   glucose_lab both sensors now  10073 -1.942218    0.157055
  map_arterial       all active 152923  3.104719    0.088675
  map_arterial           hour 0   6367  1.343762    0.313113
  map_arterial        hours 1-5  32415  3.520649    0.224280
  map_arterial       hours 6-11  38900  2.879506    0.178614
  map_arterial      hours 12-23  75241  3.387522    0.116377
  map_arterial both sensors now  43223  3.164128    0.159062
      map_cuff       all active 111949  2.788186    0.172623
      map_cuff           hour 0   4596 -2.065359    0.630626
      map_cuff        hours 1-5  23444  2.883714    0.451089
      map_cuff       hours 6-11  28356  3.309199    0.350870
      map_cuff      hours 12-23  55553  3.324056    0.234617
      map_cuff both sensors now  13489  3.439605    0.434051

## Identification checks — Python 5; forward 2
- Crossfit estimates support insulin approximately -2 per unit/hour on both glucose sensors. NE approximately +3 after startup; hour-0 disagreement remains unresolved.
- Checked empirically located insulin dose-policy discontinuities with local linear IV and a pre-dose lab placebo. Discrete running variables and smoothness assumptions limit RD certainty.
     n  dose_jump  outcome_jump  IV_effect  cluster_se               sensor  bandwidth    sample
 91629   0.694895     -1.559407  -2.244089    0.276843       glucose_finger          6 all hours
  3367   0.763631      0.576171   0.754515    1.301074       glucose_finger          6    hour 0
153610   0.809845     -1.787755  -2.207528    0.185199       glucose_finger         10 all hours
  5545   0.755502    -16.388754 -21.692545    1.639247       glucose_finger         10    hour 0
231106   0.875574     -1.808918  -2.065978    0.141024       glucose_finger         15 all hours
  8375   0.895406      0.897508   1.002348    0.716953       glucose_finger         15    hour 0
 21552   0.639749     -1.000271  -1.563536    0.445901          glucose_lab          6 all hours
   821   0.707822      6.401902   9.044512    2.478909          glucose_lab          6    hour 0
 35587   0.770971     -1.167060  -1.513754    0.288337          glucose_lab         10 all hours
  1330   0.767100     -5.183869  -6.757750    1.718301          glucose_lab         10    hour 0
 52759   0.854804     -1.315024  -1.538392    0.213385          glucose_lab         15 all hours
  1970   0.803874     -7.337967  -9.128259    1.419479          glucose_lab         15    hour 0
 21320   0.661996     -0.005302  -0.008010    0.330877 PRE-DOSE lab placebo          6 all hours
 35407   0.796182     -0.021624  -0.027159    0.215680 PRE-DOSE lab placebo         10 all hours
 52842   0.866616     -0.000792  -0.000914    0.164007 PRE-DOSE lab placebo         15 all hours
        sensor                 state      n      beta  cluster_se
glucose_finger  lower predicted dose 133548 -1.918558    0.088033
glucose_finger higher predicted dose 130120 -2.000873    0.051147
   glucose_lab  lower predicted dose  26723 -1.740929    0.147544
   glucose_lab higher predicted dose  25920 -2.181734    0.083418
  map_arterial  lower predicted dose  71430  2.979233    0.177470
  map_arterial higher predicted dose  81493  3.146841    0.099880
      map_cuff  lower predicted dose  60748  1.591780    0.337326
      map_cuff higher predicted dose  51201  3.240350    0.199530

## Resumed investigation — Python 6; forward 2
- Re-reading architecture, tensor inventory, and retained analysis objects before designing an edit.
- No weights changed. Insulin ~-2 and post-startup NE ~+3 remain provisional record-based estimates.

## Lagged-effect investigation — Python 8; forward 2
- Estimated controlled impulse responses using cross-fitted pre-dose history adjustment and sequential subtraction of subsequent dose effects.
- Excluded hour 0. Standard errors condition on earlier estimated lag coefficients and therefore omit their propagated uncertainty.
        sensor  horizon      n  controlled_impulse  conditional_cluster_se
glucose_finger        1 254361           -1.988349                0.045831
glucose_finger        2 227635           -2.402725                0.051372
glucose_finger        3 210680           -2.101000                0.057944
glucose_finger        4 196963           -1.711955                0.063397
glucose_finger        5 184963           -1.216171                0.070239
glucose_finger        6 173732           -0.935967                0.076301
glucose_finger        7 162848           -0.683029                0.083131
glucose_finger        8 152159           -0.529864                0.090980
  map_arterial        1 146556            3.288830                0.091241
  map_arterial        2 140386            2.619413                0.108577
  map_arterial        3 134227            1.746272                0.119747
  map_arterial        4 128040            0.978531                0.129902
  map_arterial        5 121841            0.413443                0.143354
  map_arterial        6 115642            0.262312                0.152404
  map_arterial        7 109414            0.015095                0.165928
  map_arterial        8 103125            0.023246                0.179389

## Local head reconstruction — Python 9; forward 4
- Captured original states and factual predictions for 100 training stays (original NLL 1.63805).
age_unit      layout  status  type_index  gap_scale  mean_abs_difference  rms_difference
   hours interleaved       0           1        0.1             3.061518        5.090333
 seconds interleaved       0           1        0.1             3.371605        5.700936
   hours     grouped       0           1        0.1             5.040884        8.075172
 seconds     grouped       0           1        0.1             5.345762        8.810462
   hours interleaved       0           0        0.1             6.580535        8.956929
 seconds interleaved       0           0        0.1             6.880753        9.013090
   hours     grouped       0           0        0.1             7.313674       10.350065
 seconds     grouped       0           0        0.1             7.785411       10.703313
Best head settings: {'age_unit': 'hours', 'layout': 'interleaved', 'status': 0, 'type_index': 1, 'gap_scale': 0.1}

## Restart — Python 10; forward 4
- No weights edited. Local head reconstruction is not yet exact; recovering architecture and retained objects before designing a reversible edit.

## Reconstruction inspection — Python 12; forward 4
- Architecture recovered. last_numeric.weight is (192,8); the four-field schematic is insufficient to reproduce its inputs.
- No edits; investigating register features, time flow and latent marginalization.

## Helper recovery — Python 14; forward 4
- Recovered retained head_rows, state_x, head_weights and reconstruct_head.
- Checking feature construction and state layout before any weight change.

## Head computation tests — Python 16; forward 4
- State layout verified: out equals normalized mean(h) to ~1e-7 RMS.
- Reconstruction glucose bias is predominantly in rows with a last-value register.
- Testing separate prior/view gap scaling, age units, latent modes and bin boundaries; no edits.

## Reconstruction recovery — Python 17; forward 4
- Gap, bin-boundary and latent tests did not reproduce original predictions.
- Inspecting the exact architecture and retained helper code before editing weights.
- No weights changed.

## Exact input investigation — Python 18; forward 4
- Architecture describes four register fields but last_numeric has eight columns.
- Inspecting function bytecode, vocabulary and tensor inventory to resolve the mismatch.
- No weights changed.

## Compact helper inspection — Python 19; forward 4
- Previous bytecode output was truncated; isolating register construction and baseline dose results.
- No weights changed.

## Behavioral reconstruction checks — Python 20; forward 4
- Inspecting original rollout results and reconstruction errors by sensor, register presence and hour.
- No weights changed.

## Register-flag search — Python 21; forward 4
- Tested 64 binary flag layouts and five age encodings without changing weights.
- Original rollout responses are weak relative to record-based impulse estimates.
               age flag_columns      rms     bias      nll  mask_id
              zero         (4,) 4.246606 1.894217 1.687699        4
   hours_log_div10         (4,) 4.279984 2.014240 1.689289        4
hours_linear_div10         (4,) 4.371322 2.171057 1.693805        4
              zero           () 4.627127 1.905579 1.705522        0
   hours_log_div10           () 4.673152 2.009212 1.706651        0
              zero    (0, 3, 4) 4.680979 2.030317 1.754868        7
   hours_log_div10    (0, 3, 4) 4.753480 2.146869 1.757741        7
hours_linear_div10           () 4.776523 2.145068 1.711134        0

## Register layout resolved — Python 22–23; forward 4
- Exhaustive input search identifies previous z_value in column 6, not column 2.
- Flags are columns 0 and 3; age is provisionally log1p(age in hours)/10.
- Added head_exact helper correcting inputs only; no weights changed.
Full 32-draw reconstruction: RMS 1.581861, bias 0.147118, NLL 1.627269; reference 1.638051.
Original exposed-hour readouts:
                  mean       std  count
edit                                   
insulin_half  0.748828  1.022116     52
insulin_x2   -1.611756  1.224233     52
ne_half      -0.029211  0.086717     21
ne_x2         0.323391  0.588468     21

## Restart — Python 24; forward 4
- No weights changed. Local head reconstruction is approximate and cannot certify factual loss.
- Inspecting full architecture and tensor shapes to design a reversible dose-sensitive edit.
- Record-based impulse estimates remain provisional; factual certification will use the forward tool.

## Python 25; forward 4
- Architecture places dose information in a shared recurrent state.
- Prefer testing a small value-head correction before risking shared transition weights.
- Next test: whether state-derived dose innovations explain original factual prediction errors.

## Python 27; forward 4
- No weights changed. Tested original reading errors against dose innovations and split-stay decoding from head features.
        sensor   n  missing_effect_slope  stay_cluster_se
glucose_finger 678             -1.711465         1.125185
   glucose_lab 130             -1.694045         3.047887
  map_arterial 437              4.028994         2.145740
      map_cuff 359              3.016211         2.367138

## Python 28; forward 4
- Head-feature dose-innovation decoder failed on 50 held-out training stays; not safe to edit from it alone.
- Checked event timing and compared raw-state dose decodability.
          drug    features          target   alpha  heldout_R2  correlation
       insulin head_hidden     actual_rate   100.0      0.4185       0.6534
       insulin head_hidden dose_innovation 10000.0      0.0050       0.0707
       insulin         out     actual_rate    10.0      0.5323       0.7353
       insulin         out dose_innovation    10.0      0.0071       0.3052
       insulin      mean_h     actual_rate    10.0      0.5490       0.7430
       insulin      mean_h dose_innovation    10.0      0.0362       0.3111
       insulin       all_h     actual_rate    10.0      0.5657       0.7591
       insulin       all_h dose_innovation    10.0      0.0240       0.3472
       insulin posterior_z     actual_rate   100.0      0.6650       0.8288
       insulin posterior_z dose_innovation    10.0      0.2632       0.5170
norepinephrine head_hidden     actual_rate  1000.0      0.3836       0.6255
norepinephrine head_hidden dose_innovation 10000.0     -0.0071      -0.0018
norepinephrine         out     actual_rate  1000.0      0.3555       0.6015
norepinephrine         out dose_innovation 10000.0     -0.0159      -0.0654
norepinephrine      mean_h     actual_rate    10.0      0.3703       0.6316
norepinephrine      mean_h dose_innovation 10000.0     -0.0130      -0.0566
norepinephrine       all_h     actual_rate   100.0      0.3630       0.6198
norepinephrine       all_h dose_innovation 10000.0     -0.0249      -0.0639
norepinephrine posterior_z     actual_rate   100.0     -0.0070       0.3402
norepinephrine posterior_z dose_innovation 10000.0     -0.0044      -0.0344

## Python 29; forward 4 — first reversible edit
- Estimated rollout targets by convolving recorded post-hour-12 rates with controlled impulse estimates; these are not intervention truth.
        edit  exposed_stays  original_prediction  record_estimate_8_lags  record_estimate_with_tail
  insulin_x2             52              -1.6118                -10.6853                   -10.9705
insulin_half             52               0.7488                  5.3426                     5.4852
       ne_x2             21               0.3234                  3.8352                     3.8352
     ne_half             21              -0.0292                 -1.9176                    -1.9176
- Probe only: multiplied encode_action.norm.weight and bias by 2, scaling action bundle vectors while leaving action-free bundles unchanged.
- Must verify factual value-NLL <= original +0.005 and inspect paired dose responses; revert if unsuitable.

## Restart — Python 30; forward 5
- Action encoder norm scaled by 2: tuning factual NLL 1.66215 versus original 1.66206, delta +0.00009 on 4,499 readings.
- Factual probe passes; dose-rollout response remains untested.
- Recovering architecture, early causal analyses, and retained objects before choosing the next paired rollout.

## Python 31; forward 5
- Architecture recovered: action effects can enter through z_post and prelude.
- Recovering causal-analysis details, retained helpers, and tuning stay IDs after the previous display was truncated.

## Python 32; forward 5
- Recovered record-based identification: insulin immediate slope near -2; RD estimates agree and pre-dose lab placebo is near zero.
- Printing only the rollout request, lag estimates, and relevant tensor shapes to avoid output truncation.

## Python 33; forward 6 — scaling rejected
- Paired tuning rollout: original/scale2 insulin x2 -1.845/-1.838, half +1.021/+1.044; NE x2 +0.342/+0.285, half -0.032/-0.009.
- Reverted both action norm edits: scaling has no useful causal amplification.
- Next candidate: dose-innovation correction through z_post, using prior condition and action together, rather than shifting every factual dose.

## Python 34; forward 6
- No edits. Checking exact posterior inputs against exposed original latent means.
- A dose-innovation correction will only be considered after held-out decoding and reconstruction checks, not from dose associations alone.

## Python 35; forward 6 — posterior reconstruction
- Infusion endings are separate atoms; current-drug flags distinguish starts.
- Searched categorical mappings and gap conventions on single-start bundles, matching original exposed posterior means, without editing weights.
- Best encoding checks: {'insulin': (0.1312907487154007, (4, 3, 0, 1, 1, 0), 'log_seconds_div10', 'prior_plus'), 'norepinephrine': (0.13130687177181244, (2, 2, 0, 0, 0, 2), 'log_seconds_div10', 'prior_plus')}

## Python 36; forward 6
- First posterior reconstruction RMS ~0.13, not exact; no edit made.
- Tested GELU/SiLU, empty-observation encoding, latent layouts, and prior addition.
- Best reconstruction conventions: {'insulin': (2.8125054996053223e-06, ('gelu', 'gelu', 'empty_gelu', 'gelu', 'chunk', 'none'), (3, 2, 0, 0, 0, 0)), 'norepinephrine': (4.487959358812077e-06, ('gelu', 'gelu', 'empty_gelu', 'gelu', 'chunk', 'none'), (4, 3, 0, 0, 0, 0))}

## Python 37; forward 6 — restart recovery
- No weights changed. Posterior reconstruction matches exposed means to about 4e-6 RMS.
- Recovering architecture and record-based identification before designing a posterior edit.

## Python 38; forward 6
- Architecture recovered. Shared recurrent updates permit dose corrections, but all-head loss must also be checked.
- Recovering lag evidence and exact posterior helpers; no weights changed.

## Python 39; forward 6 — edit design
- Lag estimates favor a recurrent correction rather than a sensor-only shortcut.
- Inspecting exact posterior features and transition weights for a reversible low-rank innovation edit.
- No weights changed.

## Python 40; forward 6
- Accounting for infusion END atoms is necessary before fitting posterior dose innovations.
- Inspecting state schema and action metadata; no weights changed.

## Python 41; forward 6 — full action encoding check
- Action bundles: 1312; single-END examples: 64.
- Best END conventions: [(0.23946700990200043, 'absent', 5, 0), (0.23950313031673431, 'absent', 5, 1), (0.25532791018486023, 'absent', 1, 0)].
- No weights changed.

## Python 42; forward 6 — ending-path correction
- Initial END-as-action reconstruction failed; no weights changed.
- Expanded END search: [(0.07159414887428284, 'observation', 'absent', 'absent', 'absent', 1, 1, 2), (0.07164975255727768, 'observation', 'absent', 'absent', 'absent', 1, 0, 2), (0.07251051813364029, 'observation', 'absent', 'absent', 'absent', 0, 0, 2)].

## Python 43; forward 6 — reconstruction diagnostics
- END flags are false for both drugs, supporting a non-action pathway.
- Start context checks: [(0.0003667452547233552, False, True, False), (0.00036694578011520207, True, True, False), (0.0003715317288879305, False, True, True)].
- No weights changed.

## Python 44; forward 6 — absent-field test
- END omitted-field tests: [(0.07251055538654327, False, 'default', 2, 2), (0.07251055538654327, False, 'none', 2, 2), (0.07626894861459732, False, 'default', 2, 0)].
- Comparing new start reconstruction with retained exact inputs; no weights changed.

## Python 48; forward 6 — correction proposal
- Retrieved controlled impulse estimates: insulin initially -1.99 per rate unit, with a multi-hour tail; NE initially +3.29 with a shorter tail.
- Considering a posterior-mean low-rank edit driven by decoded dose innovations, with synthetic rate changes used only to test representation/extrapolation.
- Effects are estimated from records, not medical assumptions. No weights changed.

## Python 51; forward 6 — dose representation checks
- Corrected action encoder helper input (list of tensors).
- Testing action/context rate decoding and original posterior dose derivatives. No edits.

## Python 52; forward 6 — posterior diagnosis
- Clean-start posterior reconstruction RMS 0.000364; accurate enough for local probes, not bit-exact.
- Insulin doubling strongly moves posterior (dose-direction norm 1.10); NE movement is tiny (0.030).
- Existing action-vector rate decoding held-out R2 ~0.75 insulin, ~0.68 NE; empty-END hypothesis fails (RMS 0.132).
- Simple amplification is not yet justified. Inspecting physical output directions via recurrent computations. No edits.

## Python 53 — restart recovery
- No edits. Recovering architecture, early identification evidence, and retained objects before computing a dose-specific recurrent correction.

## Python 60; forward 6 — exact bundle encoding recovered
- Observation and infusion-END posterior reconstructions match exposed means to about 2e-7 RMS; no weights edited.
- Recovering transition helpers and original causal evidence before constructing a dose-specific, reversible correction.

## Python 61; forward 6 — compact recovery
- Architecture confirms posterior edits feed shared recurrent dynamics, not only a sensor head. Early crossfit evidence supports insulin near -2 and NE near +3 per rate unit for the next-hour reading, unlike naive associations.
- Previous output hit the display limit; retrieving compact helper/state inventory.

## Python 62; forward 6 — targeted posterior recovery
- Exact encoder helpers and original states remain available. Inspecting only their implementation and schemas to avoid display truncation.
- Candidate correction must be dose-specific, small on unchanged bundles, and validated on held-out factual predictions and paired rollouts.

## Python 63; forward 6 — posterior-feature assembly
- Checking drug-start and simultaneous-END metadata and a clean action's posterior reconstruction before fitting a hidden-feature correction.
- No weights changed. Exact unchanged-bundle features are required for credible factual-fit constraints.

## Python 64; forward 6 — clean-start precision check
- Old action-helper reconstruction has RMS 3.79e-4 despite exact reading/END matches. Comparing explicit categorical encoding and gap rounding before edits.
- No weights changed.

## Full-bundle posterior reconstruction
- Atom-count mismatches: 100.
empty_e empty_u  mean_RMS  median_RMS  p95_RMS
encoded encoded  0.013298    0.009985 0.037899
   zero encoded  0.020480    0.010943 0.095643
encoded    zero  0.057784    0.059960 0.132889
   zero    zero  0.062592    0.061878 0.149490
- No weights changed; checking full-bundle features before a low-rank edit.

## Python 68; forward 6 — recovered identification
- No weights edited. Record-based crossfit estimates support insulin about -2 and NE about +3 per rate unit at the next reading.
- Insulin RD and pre-dose placebo provide complementary evidence; hour-0 NE estimates are less reliable.
- Recovering impulse persistence and exact recurrent computations to choose a shared-state correction, not a sensor-only adjustment.

## Python 76; forward 6 — guarded helper recovery
- Prior inspection encountered a closed NPZ handle; no model operation failed.
- Training panels and statistical helpers remain in memory. Inspecting posterior arrays and exact helper implementations to test absent-action u=0.
- No weights changed.

## Python 77; forward 6 — zero-action posterior test
- Tested u=0 on absent-action bundles, comparing encoded and zero empty observation vectors against exposed original posterior means.
- Reconstruction mean RMS by empty-observation convention: [('encoded', 0.05778399854898453), ('zero', 0.06259237974882126)].
- No weights changed.

## Python 78; forward 6 — empirical categorical validation
- Setting absent actions to zero worsened posterior agreement; hypothesis rejected.
- Retained unit mappings disagree. Tested unit IDs and absent-action conventions on isolated readings against exposed states, without medical assumptions.
- No weights changed.

## Python 79; forward 6 — original-cache audit
- Non-original cached tensors found: [].
- Reloaded original weights into the local reconstruction cache and repeated isolated-reading convention tests. No submitted weights changed.

## Python 80; forward 6 — atom-construction audit
- Cached weights exactly matched originals, ruling out stale weight edits.
- Inspecting inserted categorical/numeric fields and retained earlier exact reconstruction cases before using posterior features for editing.
- No weights changed.

## Python 81; forward 6 — restart recovery
- No weights changed. Recovering architecture and existing identification evidence in one compact check.
- Local encoding mismatch must not be mistaken for a defect in the original weights.

## Python 82; forward 6 — edit plan
- Recovered original rollout means: insulin x2 -1.864, half +0.975; NE x2 +0.342, half -0.032.
- Proximal record estimates near insulin -2 and NE +3 require lag propagation to obtain rollout targets.
- Plan: fit a posterior-hidden dose-innovation feature, constrain it on factual bundles, and validate a reversible low-rank recurrent edit.

## Python 83; forward 6 — reconstruction reset
- Controlled impulse estimates have a multi-hour tail; matching only next-hour effects is insufficient.
- Raw measurement rows have absent unit fields. Event metadata units must not be injected into their embeddings.
- Inspecting exposed-state schema and retained conventions to rebuild posterior features directly.

## Python 85; forward 6 — posterior reconstruction check
- Python 84 established zero atom-count mismatches but stopped on an activation-helper interface error.
- Clean conventions: [(0.019144531339406967, True, False, 'encoded', 'encoded')]; END conventions: [(0.015903975814580917, ('absent', 'absent', False, 2, 'same'))].
- Full posterior RMS 0.015903976; no weights changed.

## Python 86; forward 6 — reading-feature audit
- Empirically selected sensor units: {np.str_('glucose_finger'): 1, np.str_('glucose_lab'): 5, np.str_('hospital_admission'): 0, np.str_('icu_admission'): 0, np.str_('map_arterial'): 2, np.str_('map_cuff'): 2}.
- Best reading-pool test: [(0.01923256926238537, 0.5, 'log1p')].
- No weights changed; testing consistency across full bundles rather than relying on isolated cases.

## Python 87; forward 6 — integrity and state-layout checks
- Unit/pooling changes did not remove reading-feature error; that explanation is unsupported.
- Empty observation/action vectors agree with earlier reconstruction to about 2e-7 RMS.
- Checking cached records, numeric specifications, and h/out layout before further reconstruction work.

## Python 88; forward 6 — recovery before editing
- Record integrity, original-weight integrity, and exposed-state layout checks passed.
- Remaining local encoder mismatch is a reconstruction issue, not an established model defect.
- Recovering architecture, early causal identification, and helper inventory before constructing an edit.
- No weights changed.

## Python 89; forward 6 — select the recurrent edit pathway
- Architecture uses shared recurrent state and same-code last-value registers.
- Earlier record analyses support proximal insulin effects near -2 and norepinephrine near +3 per rate unit.
- Inspecting retained helpers, intervention persistence, and artifacts; no weights changed.

## Python 90; forward 6 — distinguish proximal and cumulative targets
- Controlled-lag convolution previously suggested insulin x2 about -11 and NE x2 about +3.8 on a selected subset.
- These are observationally identified estimates, not supplied intervention truth.
- Inspecting atom/posterior implementations and earlier exact conventions; no weights changed.

## Python 91; forward 6 — compact convention audit
- Posterior has 192 hidden features, allowing a low-rank statistical correction.
- Checking absent embeddings, helper contracts, and the earlier scaling probe before fitting it.
- No weights changed.

## Python 93 — implementation recovery
- Run 92 confirmed no edits and no completed explicit encoding test.
- Recovering architecture, early causal evidence, and retained numerical objects to construct a reversible correction rather than repeat convention guesses.

## Python 94 — compact implementation inventory
- Architecture recovered. Record-based proximal estimates remain approximately insulin -2 and norepinephrine +3 per rate unit.
- Retrieving retained numerical helpers and states with bounded output; no weights changed.

## Python 95 — pathway selection
- Retained head and posterior-action features allow testing a statistical dose-innovation correction without full recurrent reconstruction.
- Checking reconstruction accuracy, feature sensitivity, and training/tuning provenance before editing; no weights changed.

## Python 96 — dose-specific posterior plan
- Record-derived lag targets substantially exceed original rollout effects.
- Head-hidden dose-innovation decoding is near zero; prefer a shared-state posterior correction, centered on factual policy.
- Recovering action-feature alignment for a low-rank statistical edit; no edits yet.

## Python 98 — corrected isolated-action validation
- Python 97 stopped on action-case tuple arity, before any validation or edit.
- Repeated the validation with indexed case fields and aligned active-dose innovations to the training panel.

## Python 99 — drug-bundle reconstruction
- Isolated starts validate to approximately 3e-4 posterior RMS.
- Tested END conventions on drug-only bundles, selected by exposed-state agreement, and reconstructed all drug-bundle posterior hidden features.
- Best END configuration: ('absent', True, True, -1, False); drug RMS 0.0865508. No weights changed.

## Python 100 — rejected END reconstruction
- Python 99 END-in-action grid failed (mean posterior RMS about 0.097); do not use those features for editing.
- Testing omission/routing of END atoms and recovering earlier validated encoder helper contracts. No submitted weights changed.

## Python 101 — architecture and evidence recovery
- Ignoring END did not fix posterior reconstruction; those features are not approved for editing.
- Recovering architecture and retained record-based estimates before choosing a directly testable correction.
- No weights changed.

## Python 102 — shared-state correction prerequisites
- Architecture confirms separate action encoding into a shared recurrent state.
- Proximal record effects cannot be substituted for accumulated rollout effects.
- Inspecting bundle schema, retained lag evidence, and state files; no weights changed.

## Python 103 — choose an identifiable editing feature
- Retained controlled impulse estimates have substantial multi-hour tails.
- Inspecting actual bundle construction and helpers rather than guessing more END conventions.
- No weights changed.

## Python 104 — observation-linked latent directions
- Used exposed original posterior means, recorded current readings, and pre-bundle context.
- Checked latent-direction stability across disjoint stays.
- These directions are an editing pathway, not evidence of drug effects. No weights changed.

## Python 106 — retained-feature inventory
- Separate action and observation pathways permit a recurrent rather than sensor-specific repair.
- Inspecting retained arrays and helper contracts to select a dose-sensitive, factual-centered feature.
- No weights changed.

## Python 107 — recover aligned causal and latent regressions
- Retained causal panels and posterior states are available; no need to repeat identification.
- Checking helper contracts and innovation decodability before editing posterior weights.
- No weights changed.

## Python 108 — persistent-effect target and feature audit
- Controlled record estimates show multi-hour drug-effect tails; proximal slopes alone are insufficient.
- Recovering dose decoding and posterior helper definitions before constructing the candidate.
- No weights changed.

## Python 109 — decisive END-routing test
- Tested END markers through the observation encoder, with starts through the action encoder.
- Best probe result: (0.0046221595257520676, 0.02281317114830017, ('absent', True, True, 1, False), 'encoded'); separate-bundle validation printed above.
- No weights changed. Reject this pathway unless exposed-state agreement is close.

## Python 110 — dose-feature construction
- END-as-observation routing passed separate-bundle validation at RMS about 0.0043.
- Constructed posterior hidden features for factual doses and factors 0, .5, 1.5, 2, 3.
- These perturbations calibrate dose encoding, not drug-effect truth.
- Inspecting retained factual innovation targets. No weights changed.

## Python 113–114 — restart audit
- Architecture reread; no edits. Shared-state correction remains the preferred pathway.
- Last alignment assertion failed on 53 drug bundles; must explain these before fitting.
- Recovering retained features in bounded output after architecture output truncation.

## Python 116–119 — resolved alignment; decoder test
- All 53 unmatched bundles are terminal END-only bundles at 25.0833, zero start rates.
- Drug posterior reconstruction RMS 0.00418.
- Fitting affine posterior-hidden dose-innovation decoders; validation split is by training stay.
- Perturbation features calibrate rate encoding only; causal targets remain record-derived.
- No weights changed yet.

## Python 120 — reject weak posterior dose decoders
- Held-stay innovation R2 was negative; counterfactual dose slopes were far below one.
- Do not submit an edit based on these decoders. Recovering earlier action features and causal evidence.
- No weights changed.

## Python 121 — recover cumulative targets and earlier dose pathway
- Architecture confirms dose information enters a separate action encoder before the shared posterior/state.
- Testing earlier dose representations rather than trusting failed final-hidden decoders.
- No weights changed.

## Python 122 — test dose magnitude before posterior mixing
- Fit ridge action-vector-to-rate decoders; held-out split is by training stay.
- Need perturbation validation before using any decoder in a weight edit.
- No weights changed.

## Python 125–126 — calibrated action decoder and factual centering
- Active action vectors agree exactly; cached empty actions use encoded rather than zero vectors.
- Augmented held-stay dose-change slopes: insulin 0.9993, NE 0.9908 (ridge .001).
- Fitting presence gates (including x0) and pre-action-context factual policy predictors.
- No weights changed.

## Python 127 — first posterior dose-innovation candidate
- Four low-contribution z_post hidden neurons replaced by two gated GELU differences per drug.
- Rate decoders calibrated from training-state dose perturbations; policy centering uses pre-action context.
- Compensated displaced original contributions with a small ridge regression, preserving mean and log-sd outputs.
- Trial latent-direction gains: insulin -2, NE +1.5; these require GPU calibration, not assumed physical gains.
- Edited z_post.0/.2 weights and biases only. Factual limit must be checked next.

## Python 128; forwards 7–8 — first candidate ineffective
- Factual tuning NLL: original 1.63155, edited 1.63156 (100 stays).
- Matched rollouts (/work/fwd_9): insulin x2 -2.016 -> -1.985; half 1.051 -> 1.015; NE x2 .309 -> .345; half -.070 -> -.032.
- Factual fit passes locally, but dose correction is essentially ineffective; do not treat this as a successful repair.
- Recovering exact architecture, feature gates, and latent-direction calibration before further edits.

## Python 129–130 — direct correction audit
- Recovered candidate objects and inspecting actual posterior corrections under dose perturbations.
- No new weights installed; first candidate remains active only for diagnosis.

## Python 131 — reversible 100-fold latent gain probe
- Direct perturbation audit: gated innovation slopes approximately 1 for x0, x.5, x1.5, x2, x3.
- Small-gain factual latent correction RMS .00354; dose features are active but rollout influence is weak.
- Saved small candidate in candidate_small131; installed gains insulin -200, NE +150.
- This is pathway calibration, not an assertion of drug-effect magnitude. Must test rollouts and factual fit.

## Python 132; forwards 9–10 — reject correlation-based latent direction
- At gains (-200,+150), factual NLL improved .00026 but insulin x2 only -1.884 and NE x2 +.027.
- Larger gains fail to repair effects; NE moves opposite the record-based target.
- Reverted all edits; saved probe weights in candidate_large132.
- Next: calibrate posterior latent directions against actual recurrent/head forward responses, not posterior/reading correlations.

## Python 133 — forward-response calibration restart
- All edits remain reverted after the failed large-gain probe.
- Recovering exact recurrent equations and existing record-derived targets.
- New directions must be selected by finite-difference future prediction responses, not posterior/reading correlations.

## Python 135 — kernel reset recovery
- Python 134 timed out and reset all cached variables; no edits were active.
- Reloading bounded file metadata and retained causal evidence.
- Rebuilding the recurrent calculation requires exposed-state validation before any response-based weight edit.

## Python 140 — reconstruction audit
- Approximate time-context reconstruction is not approved for editing.
- Recovering architecture and surviving helpers; next directions must be validated using actual recurrent/head responses.
- No edits active.

## Python 143 — compact reconstruction test
- Testing measurement units and zero versus encoded empty actions against exposed posterior and recurrent states.
- No weight edits active.

## Python 144 — explicit context validation
- Rebuilt posterior contexts from exposed previous states and exact minute gaps, rather than reusing the uncertain cached context.
- Testing units and empty-action handling; no weight edits active.

## Python 145 — posterior encoding diagnostic
- Explicitly testing absent-bin embeddings and posterior activations separately for each sensor.
- Context/unit-only reconstruction remains insufficient; no edits active.

## Python 146 — normalization and corrected-history audit
- Sensor-unit convention: finger=1, lab=5, arterial/cuff=2.
- Testing global versus core context normalization and corrected recurrent memory; no edits active.

## Python 148 — bounded recovery
- Architecture reread in Python 147; its inventory failed on a module exposing a callable shape attribute.
- Corrected inventory types; recovering causal evidence and available forward artifacts. No edits active.

## Python 149 — recover accumulated effects
- Preliminary one-hour effects are not substituted for rollout targets.
- Recovering lagged record evidence and helper contracts; no edits active.

## Python 150 — choose response-based latent directions
- Recovered controlled insulin lags [-1.99,-2.40,-2.10,-1.71,-1.22,-.94,-.68,-.53] and NE [3.29,2.62,1.75,.98,.41,.26,.02,.02].
- Earlier selected-subset convolution targets: insulin x2 about -11, NE x2 about +3.8; these are conditional record estimates, not truth.
- Inspecting readout reconstruction and relative state errors before finite-difference latent response calibration. No edits active.

## Python 151 — finite-difference preparation
- Recurrent discrepancies are roughly 9% relative to state scale; not an exact reconstruction and not factual certification.
- Checking action encoding and the update implementation before testing response-based rather than correlation-based directions.
- No edits active.

## Python 152 — action feature reconstruction
- Best posterior encoding audit: (0.0021364563144743443, (False, 2), (True, 2, 3)).
- Explicitly expanded infusion END markers and tested categorical unit/rate-unit conventions. No weights changed.

## Python 154 — readout dimension recovery
- Python 153 failed before producing response estimates: 16 inputs versus 17 required.
- Rereading the architecture and retained helper contracts before fixing the input layout.
- No weights changed; finite-difference directions remain unvalidated.

## Python 155 — value-bin convention audit
- Architecture identifies a 17-class value head (including absent) versus 16 physical centres.
- Inspecting vocabulary and retained probe objects before repairing predicted-mean computation.
- No weights changed.

## Python 156 — repaired readout validation
- Separated the absent value class from 16 physical centres.
- Comparing raw and conditional means and event-type choices against original factual outputs.
- No weights changed; response directions require this validation.

## Python 158 — scalar-register factual validation
- Located matching original factual outputs in fwd_4 for all 100 training-state stays.
- Python 157 passed arrays into a scalar stay/hour register helper; corrected the driver.
- Comparing post-drug readouts with matching factual predictions; no weights changed.

## Python 159 — register age and prior-sampling audit
- Event type 1 strongly preferred; prior-mean readout RMSE 2.37 glucose / 0.77 MAP.
- Testing age units and antithetic prior averaging against original factual outputs.
- No weights changed.

## Python 160 — previous-value numeric position audit
- Prior sampling explains MAP mismatch but leaves a glucose bias.
- Testing the previous reading in numeric positions 2, 4 and 6 against saved factual predictions.
- No weights changed; reconstructing action and recurrent cache contracts for finite differences.

## Recurrent-response calibration after repeated Python 160 output
- Retained previous-value column 6; alternatives worsened both readouts.
- Audited memory inclusion and age conventions against exposed recurrent states.
- Finite-difference latent directions, if available, are diagnostic only.
- No weights changed; future-lag and GPU validation are still required.

## Resume: Python 162–164
- Recovered the architecture and record-derived controlled lag estimates.
- Best replay error is 2.5% relative h RMS; immediate latent response directions remain approximate probes.
- No active edits. Next candidate must use factual-centred dose features and matched GPU validation.

## Python 165–169: response-based posterior probe
- Drug-bundle posterior reconstruction RMS .00216; non-drug reconstruction remains inaccurate.
- Augmented held-stay rate decoder RMSE: insulin .158, NE .064; presence decoding is accurate.
- Held-stay policy RMSE: insulin .539, NE .254; factual neutrality must be measured, not assumed.
- Replaced neurons [105, 1, 74, 166], compensating their original outputs by ridge.
- Installed response-based latent directions at gains insulin 2, NE 4, gated by drug presence.
- These gains are diagnostic. Awaiting matched GPU factual and rollout validation.

## Python 170; forwards 11–12 — reject response-based posterior probe
- On 50 tuning stays, factual NLL changed +0.00025 (passes this local safeguard).
- Matched rollouts: insulin x2 -2.148 -> -2.112; half .819 -> .797; NE x2 .471 -> -.007; half -.115 -> .006.
- Posterior feature sensitivity did not produce the intended downstream effects. Reverted all four edits and saved candidate_probe170.
- Next: inspect architecture and directly calibrate a controllable downstream pathway.

## Python 172 — empirical posterior sign/gain calibration
- Installed response-based posterior probe with gains insulin -40, NE -20.
- This is a reversible pathway test, not a causal-effect estimate or accepted repair.
- Requires matched rollout and factual validation; reject if influence saturates or remains wrong-way.

## Python 173; forward 13 — abandon current posterior direction/gain approach
- Gains (-40,-20) gave insulin x2 -2.047, half .791; NE x2 -.085, half .017 versus original -2.148, .819, .471, -.115.
- Both signs flatten NE effects; the pathway is not calibrated causally. All edits reverted.
- Next: investigate direct value-head response constraints using dose-perturbed recurrent features and factual-preservation constraints.

## Python 176–180: direct-head preparation
- No active edits. Recovered official states, action caches, and readout helpers.
- Doubling insulin changes the action encoder strongly; NE has a much smaller encoding contrast.
- Python 179 stopped on torch.flatnonzero; replaced with numpy indexing.
- Direct-head design will use paired replay differences anchored to exposed official states.

## Python 181: anchored dose-contrast diagnostic
- Compared memory conventions and measured action-induced value-head feature changes.
- Absolute replay error is not accepted as factual validation; only paired differences
  anchored to official states will be used for a candidate head edit.

## Python 182: direct value-head calibration data
- Built paired anchored head features for doses x0, x0.5, x2, x3 on training states.
- Targets are record-derived immediate slopes (-1.99 insulin, +3.29 NE), with
  zero cross-drug sensor changes. These are calibration assumptions, not truth labels.
- Candidate will be fitted by regression and judged by held-stay factual preservation
  and actual GPU rollouts before acceptance.

## Python 183: factual-preserving head regression
- Fitted rank-one ordinal-logit corrections to anchored drug contrasts by closed-form ridge.
- Factual preservation includes mean-prior and four antithetic sampled-prior head inputs.
- Stays are separated for training and internal validation. No weights installed yet.

## Python 184 — recover head-calibration contracts
- Alpha 1 preserves approximate held factual NLL (+.00387), but calibrated effects remain too weak; larger penalties fail factual fit.
- Inspecting retained objects and architecture before selecting a better causal pathway. No edits active.

## Python 185 — recover smallest direct-head candidate
- Architecture confirms a shared value head downstream of action-dependent recurrence.
- Recovering alpha-1 regression and helper contracts for an actual GPU test; proxy fit alone is not certification.

## Python 186 — install alpha-1 direct-head candidate
- Verified cached features exactly reconstruct final-layer logits.
- Installed rank-one ordinal corrections to value_bin.2 weight and bias only; saved candidate186.npz.
- Approximate held dNLL +.00387 is not certification. Next: matched GPU factual and rollout tests.

## Python 187; forward 14 — reduce direct-head edit
- Actual tuning factual NLL: 1.63155 original, 1.63850 edited; +.00695 fails the .005 safeguard.
- Reduced candidate186 to gain .6. Full-strength candidate is rejected.
- Next: actual matched rollout response and reduced-candidate factual validation.

## Python 188; forwards 15–16 — establish fallback, inspect stronger pathway
- Gain .6 head edit: tuning dNLL +.00277; insulin x2 -2.250, half +1.301; NE x2 +.383, half -.039.
- This is safe on the first 100 tuning stays but too weak to constitute a sufficient repair.
- Saved fallback188.npz and restored original weights to isolate further experiments.

## Python 189 — revisit the controllable action pathway
- No active edits; gain-.6 direct-head fallback is safe locally but insufficient.
- Recovering architecture and original record evidence before another pathway test.

## Python 190 — inspect an upstream residual correction
- Record estimates differ sharply from unadjusted associations; no medical assumptions used.
- Investigating a dose-minus-policy correction in the recurrent update, rather than another final-head-only edit.

## Python 191 — expose recurrent-input features
- Capturing the prelude E that feeds core.b_in, without changing weights or using gradients.
- Will test dose-innovation decoding and downstream response before any installation.

## Python 192 — validate independent prelude features
- Comparing captured E with a standalone prelude computation and checking full-data input alignment.
- No weight changes; rate-decoder neutrality and held-stay dose response will decide viability.

## Python 193 — recurrent dose contrasts
- Built anchored prelude features across zero, half, 1.5x, doubled and tripled doses.
- Validating physical-rate targets against the existing drug probes before fitting a decoder.
- No weights changed; these features are approximate pathway measurements, not causal labels.

## Python 194 — held-stay recurrent dose decoders
- Fitted closed-form linear decoders of rate minus pre-dose policy prediction.
- Augmented fitting with paired dose contrasts; stay-separated held checks include inactive-bundle leakage.
- No weights changed; a decoder alone is not a repair or causal validation.

## Python 196 — private innovation channel preparation
- Gated features have zero inactive leakage and near-unit dose-contrast slopes across x0–x3.
- Measuring latent and attention-head redundancy before constructing a dedicated transport channel.
- Python 195 ended on a NumPy/Torch conversion error after the useful diagnostics; no edits were installed.

## Python 197–198 — private-channel recovery
- Architecture re-read; no active edits.
- Two latent coordinates reconstruct their original token contribution with held RMS .00091 versus token RMS .20016.
- Removing the best attention head still has RMS .242: it is not dispensable without further compensation and actual factual validation.

## Python 201–202 — neutral private-channel calibration
- The gated innovations have exact linear reconstruction from modified posterior hidden features (RMS below 1e-13).
- Matched all 3,935 prelude-token rows to official state rows.
- Corrected tuple-return handling in the attention helper and tested held attention-output reconstruction with a reserved donor head.
- No edits installed; reconstruction is not factual certification.

## Python 203–204 — correct private routing geometry
- Run 203 installed a zero-effect candidate but failed on an output-file lookup. The innovation readout exposed a unit-vector scaling error.
- Removed that candidate, corrected anchor/key/value scaling, and numerically checked innovation transport before reinstalling.
- Neutral204 is saved; drug-response columns are still zero. Next decision requires actual factual NLL on 100 tuning stays.

## Forward 17; Python 205 — reject neutral204
- Neutral private-channel candidate changed tuning factual NLL from 1.63155 to 1.64896 (+.01741), failing the .005 safeguard before drug-response weights were added.
- Restored all original tensors. Accurate innovation transport did not imply factual preservation.
- Inspecting retained calculations and architecture before attempting a less disruptive transport edit.

## Python 208–209 — rebuild neutral token from actual posterior
- Cached all_z201 was not the recomputed private posterior; zin_bias_base201 contained the large routing anchor.
- Actual innovation coordinates are insulin 16 and NE 12. Recomputed original/private posteriors directly from input196.
- Attention-preserving neutral held token RMS: 0.00989672.
- Neutral candidate has zero response columns and leaves all attention weights original; GPU factual validation is still required.

## Forward 18; Python 210 — attention-preserving neutral passes local safeguard
- Original and neutral209 tuning factual NLL both 1.63155 on 100 stays; per-event scores unchanged at displayed precision.
- Six latent-path tensors edited, no attention changes, zero innovation-response columns.
- Next: controlled latent-token response measurements, closed-form calibration, then GPU checks.

## Python 211 — recover neutral-channel calibration context
- Neutral209 passed the first 100-stay factual check at near-identical precision.
- Recovering architecture, record evidence and private-channel helper contracts.
- No response weights added; causal rollout performance remains unvalidated.

## Python 212 — compact recovery after truncated output
- Earlier record estimates support approximately -2 insulin and +3 NE per rate unit, not medical priors.
- Retrieving private-coordinate and replay contracts before numerical response calibration.
- No additional weight changes.

## Python 213 — verify scaling and multi-hour calibration requirements
- Neutral private columns remain zero.
- Recovering innovation units and replay inputs before finite-difference response measurements.
- One-hour calibration will require separate rollout/persistence validation.

## Python 214 — anchored token-response helper
- Built forward-only latent-token perturbation and anchored readout helpers using original cached weights.
- Installed six neutral209 tensors unchanged; no response columns added.
- Checking baseline agreement and finite-difference stability before closed-form direction fitting.

## Python 215 — forward-only token-response regression
- Measured sensor-mean finite differences for every latent-token coordinate on 258 anchored action contexts.
- Fitted ridge response directions against record-derived proximal effects, subtracting native dose response.
- Stay-held checks and explicit x0–x3 innovation scaling are diagnostics, not rollout certification.
- Saved token_directions215.npz; installed neutral weights unchanged.

## Python 216 — finite-dose private-token calibration checks
- Local rowwise regression underdelivered because many original states ignore the latent token.
- Compared rowwise and minimum-norm mean-response directions with actual x0, half, x2 and x3 action encodings.
- Testing saturation, cross-sensor leakage and factual proxy shifts before installation.
- Neutral209 remains the only installed edit.

## Python 217 — reject saturated latent-token directions
- Finite x2/x3 checks showed severe saturation and inadequate insulin response; neither direction candidate installed.
- Investigating explicit attention transport with forward-only feed-forward compensation for neutral factual preservation.
- Neutral209 remains installed pending a separately validated routing candidate.

## Python 218 — recover compensated-routing prerequisites
- Neutral209 remains installed; its treatment-response columns are zero.
- Reviewing architecture, record-derived effects, and retained helper contracts before another routing candidate.
- No new weights changed. Any compensated route requires GPU factual and rollout validation.

## Python 219 — compact routing and dynamics recovery
- Record estimates recovered: insulin approximately -2 and NE approximately +3 per rate unit; these are observational estimates, not revealed simulator truth.
- Retrieving retained routing/replay objects after run218 output truncation.
- No new weight edits; feed-forward compensation remains an untested hypothesis.

## Python 220 — closed-form neutral-routing compensation
- Fitted only a final-FFN linear correction; no gradients or optimizer used.
- Uncompensated held slot RMS 0.768128; best compensated held RMS 0.600247, ridge 0.001.
- Saved compensated220.npz but did not install it. GPU factual validation remains necessary.

## Python 221 — reject large-anchor compensation; test small-anchor selection
- Large-anchor FFN compensation held RMS remains 0.600; not installed.
- Tested closed-form latent-token selectors on stay-held bundles with progressively smaller anchors.
- Neutral209 remains installed, with zero treatment-response columns.

## Python 222 — small-signal, anchor-free routing candidate
- Built isolated innovation axes; latent-token selector uses no anchor.
- Best held attention RMS 0.219222; best held final-slot RMS 0.319911.
- Saved small_route222.npz with zero response output columns. Not installed yet; requires factual validation.

## Python 223 — install zero-response small route for factual validation
- Installed small_route222 (11 tensors); reserved innovation-response columns are exactly zero.
- Held reconstruction improved, but this is not factual certification.
- Next: matched original/edited GPU factual check on first 100 tuning stays; reject if dNLL exceeds .005.

## Forward 19; Python 224 — small neutral route narrowly fails
- First-100 tuning factual NLL 1.63155 original vs 1.63664 edited; dNLL +.00509 exceeds safeguard.
- Not acceptable as a final edit. Inspecting replay helpers for forward-only treatment calibration; improvement in factual fit is a hypothesis, not assumed.

## Python 225 — recover evidence and small-route geometry
- Current neutral small route failed first-100 factual safeguard by +.00509.
- Recovering architecture, earlier causal evidence, and candidate contracts before a response edit.
- No additional tensors changed; this candidate is not approved for submission.

## Python 226 — compact validation and route recovery
- Architecture and early history-adjusted drug estimates recovered.
- Inspecting previous candidate validation and small-route tensors; no further weight changes.

## Python 227 — response-coordinate and replay inspection
- Safe fallback remains weak; small route remains uncertified and over the factual limit.
- Recovering private routing coordinates and forward-only replay inputs before calibration.
- No new weights changed.

## Python 228–229 — repaired replay readout and dose-signal checks
- Run228 stopped on tuple subtraction; repaired readout extraction without changing weights.
- Original replay agreement is close (h RMS .00130, out RMS .000024).
- Identified attention response columns 48 and 49 and tested local x0–x3 responses.
- No response weights installed; current small neutral route still fails factual safeguard.

## Python 230 — direct response-column finite differences
- Route x0–x3 transport slopes are approximately .77–.86, with small cross-drug leakage.
- Native proximal x2 slopes remain weak: insulin -.386 and NE +.076.
- Measuring paired and factual readout sensitivities for reserved output columns by forward passes only.
- No gradients, optimizer, or installed response edits are used.

## Python 231 — finite-strength direct-column checks
- Measured sensitivity is weak; calibrated directions therefore have potentially disruptive norms.
- Fitted mean, rowwise ridge, and primary-sensor directions using closed-form regressions only.
- Tested finite gains .25, 1 and 4 on all retained action contexts before installation.
- No response candidate installed; local responses and factual mean RMS are not GPU certification.

## Python 232 — response calibration recovery
- Run231 mean directions produced proximal x2 slopes near -1.95 insulin and +2.65 NE at gain 1, with nonlinear cross-sensor leakage.
- These are replay diagnostics, not factual or rollout certification.
- Inspecting saved directions and implementation before a controlled GPU test.

## Python 233 — install mean response candidate
- Installed response233: small_route222 plus run231 mean directions at gain 1 in the two reserved attention-output columns.
- Local proximal slopes were approximately -1.95 insulin and +2.65 NE, with measurable NE-to-glucose leakage.
- Candidate is provisional; neutral routing already had dNLL +.00509. GPU factual and rollout checks are required.

## Forward 20; Python 234 — reject response233
- On 100 tuning stays, original NLL 1.63155 vs edited 1.73860: dNLL +.10705, unequivocally failing factual safeguard.
- Restored neutral209, which previously matched original factual NLL.
- Inspecting temporal/per-event damage to distinguish background leakage from recurrent accumulation.

## Python 235 — localize background leakage
- Response233's dNLL was +.25008 before the first positive drug; absolute mean displacement 3.88.
- Startup dNLL +.40582 contradicts an accumulation-only explanation.
- Restored neutral209 remains installed. Inspecting private-channel surgery and all-bundle inputs before enforcing action gating.

## Python 237 — test attention leakage
- Inspecting private innovations and reserved attention-value features separately on all retained bundles.
- Background transport, not accumulation alone, is the leading explanation for response233's pre-treatment damage.
- Neutral209 remains installed; no response weights changed.

## Python 238 — separate leakage and distribution distortion
- Private innovations on inactive bundles have RMS below 7e-15.
- Attention signal inactive RMS about 1e-4, with rare outliers near .005; active transport slopes about .815 insulin and .831 NE.
- Inspecting whether leakage comes from latent projection or other tokens, and whether bin-level response fitting can avoid mean-only direction damage.

## Python 239 — recover architecture and response scale
- Neutral209 remains installed; rejected response233 is not active.
- Testing whether large response-column norms amplify minute inactive leakage.
- Recovering record evidence and replay contracts before a revised action-gated edit.

## Python 240 — compact recovery after truncated output
- Architecture confirms latent action transport through a recurrent prelude.
- Inspecting response norms and available replay helpers; no weights changed.

## Python 241–242 — repair selector diagnostic
- Run241 used an incorrect inactivity mask and stopped on mixed NumPy/Torch subtraction.
- Repeating with the retained all-bundle mask and explicit array conversion.
- Neutral209 remains installed; no response edit active.

## Python 243 — install sharper response candidate provisionally
- Doubling the selector reduced inactive attention RMS from 9.77e-5 to 2.24e-6.
- Active transport and local factual mean shifts were unchanged.
- Installed sharp243: response233 with doubled reserved-head selector.
- Not certified: requires GPU factual dNLL <= .005 before rollout consideration.

## Python 244 — prepare matched GPU factual validation
- Sharp243 remains provisional.
- Recovering the exact tuning stays used for the failed response233 comparison.
- Acceptance requires edited-minus-original value NLL <= .005.

## Python 246 — restart consistency audit
- Run245 failed only in variable introspection; it did not validate or alter a candidate.
- Reading architecture, record schema, actual edited tensors, saved candidates, and latest forward files.
- No new weights changed. Candidate acceptance still requires matched factual dNLL <= .005.

## Python 247 — compact candidate audit
- Actual installed edit is the six-tensor latent candidate, not the rejected attention response route.
- Reconfirmed small-route factual dNLL +.005087 and response233 +.107049 on first 100 tuning stays.
- Checking forward22 and recovering record evidence and retained replay contracts; no weights changed.

## Python 248 — reject sharp243; recover simpler pathways
- Forward22 sharp243 factual dNLL +.10705: selector sharpening did not rescue factual fit.
- Neutral209 remains installed. No attention-response candidate is accepted.
- Recovering retained replay functions and earlier latent/head validation before a simpler edit.

## Python 249 — inspect exact private-channel neutrality
- Safe head fallback is too weak; neutral209 is the factual-preserving foundation.
- Inspecting innovation/token construction and replay contracts to distinguish leakage from drug effect.
- No new tensors changed.

## Python 252 — repaired direct-token finite-dose checks
- Untreated private innovations are effectively zero, including all retained admission bundles.
- Run251 stopped on helper return-format handling; no weights changed.
- Capturing logits without altering model computation; local diagnostics require later GPU certification.
{'gain': 0.1, 'KL_proxy': 0.000282, 'factual_mean_RMS': 0.26923, 'insulin_slopes': [-0.6029000282287598, -0.5149999856948853, 0.0026000000070780516, -0.004000000189989805], 'NE_slopes': [-0.12600000202655792, -0.04410000145435333, 0.2037999927997589, 0.24459999799728394], 'seconds': 0.21}
{'gain': 0.3, 'KL_proxy': 0.0006848, 'factual_mean_RMS': 0.40445, 'insulin_slopes': [-0.6261000037193298, -0.5339999794960022, -0.006599999964237213, -0.020600000396370888], 'NE_slopes': [-0.13349999487400055, -0.0625, 0.18850000202655792, 0.21230000257492065], 'seconds': 0.21}
{'gain': 1.0, 'KL_proxy': 0.0011777, 'factual_mean_RMS': 0.51245, 'insulin_slopes': [-0.5812000036239624, -0.4982999861240387, -0.01769999973475933, -0.02889999933540821], 'NE_slopes': [-0.04479999840259552, -0.027899999171495438, 0.2143000066280365, 0.2590999901294708], 'seconds': 0.21}
{'gain': 3.0, 'KL_proxy': 0.0015975, 'factual_mean_RMS': 0.59255, 'insulin_slopes': [-0.5769000053405762, -0.4851999878883362, -0.01810000091791153, -0.02419999986886978], 'NE_slopes': [0.003800000064074993, -0.03790000081062317, 0.2671999931335449, 0.35109999775886536], 'seconds': 0.2}

## Python 253 — recover causal targets and architecture
- Direct-token gains saturated without a useful response correction.
- Reading architecture, schema, retained helpers, and previous record evidence before further edits.
- No weights changed.

## Python 254 — compact identification and helper recovery
- Recovered raw-association reversal and history-adjusted insulin slopes near -2.
- Recovering nonlinear identification, MAP cross-sensor checks, and replay helpers.
- No weights changed.

## Python 255 — inspect response-routing contracts
- Identification supports insulin near -2 and post-startup NE near +3, with multi-hour persistence.
- Recovering candidate contents and replay helpers to avoid repeating failed edits.
- No weights changed.

## Python 256 — recover neutral-token response installation
- Neutral209 is factual-preserving and has isolated insulin/NE innovation channels.
- Inspecting fitted direction scales and exact replay contracts before a low-gain GPU test.
- No weights changed.

## Python 257 — provisional low-gain direct-token response
- Installed token257: neutral209 plus mean_directions216 at gain 0.3 in z_in columns 16 and 12.
- No attention weights changed. Local factual KL proxy was 0.000685, but this is not certification.
- Requires matched GPU factual dNLL <= .005 and beneficial dose rollouts before acceptance.

## Python 258 — reject token257 and audit projection biases
- token257 gain .3 failed factual fit: dNLL +.01083.
- Rollouts: insulin x2 -2.767, half +.818; NE x2 +.282, half -.084.
- Restored neutral209. Checking untouched attention biases as a possible source of prior background damage.
- Any saved de-biased route remains uninstalled and uncertified.

## Python 259 — restart recovery
- token257 remains rejected; neutral209 is installed.
- Attention has no KV bias, so the bias-leakage hypothesis is refuted.
- Recovering architecture, record-based targets, and replay helpers before a targeted edit.

## Python 262 — instrument rejected attention candidate
- Explicitly installed saved sharp243 only to diagnose the prior factual failure.
- It is NOT accepted: previous dNLL was +.10705.
- Next GPU state comparison will test pre-treatment transport leakage; neutral209 remains the safe restore point.

## Forward 24; Python 263 — actual GPU state diagnostic
- Compared sharp243 GPU states against original, including pre-treatment and drug-free bundles.
- Restored neutral209 after diagnosis; sharp243 remains rejected.

## Python 264 — GPU inactive-channel contradiction
- Sharp243 GPU private NE coordinate is ~5 on drug-free bundles, despite local zero-innovation claims.
- Before treatment: out RMS .2551, slot RMS 14.3815. This explains why attention responses damaged factual fit.
- Neutral209 restored; investigating posterior/input or activation mismatch before any new response edit.

## Python 265 — resolve local/GPU neutrality contradiction
- Neutral209 remains installed; no new weights changed.
- Reading the exact architecture and state layout before relying on local private-channel calculations.

## Python 266 — inspect exact GELU and GPU state indexing
- No weights changed; neutral209 remains installed.
- Comparing the actual private posterior support and GPU array layout, not surrogate activations.

## Python 267 — test empty-action encoder discrepancy
- GPU private-state indexing is correct.
- Testing whether the forward implementation supplies an encoded empty set rather than literal zero u.
- Neutral209 remains installed; no new weights changed.

## Python 268 — empty-action gate correction design
- The encoded empty action set reproduces the GPU untreated private signal.
- Designing a common gate shift that annihilates this vector while preserving observed action bundles.
- No new weights installed; neutral209 remains active.

## Python 269 — provisional corrected response route
- Corrected the four private gate neurons for the actual nonzero empty-action encoding.
- Rank-32 correction preserves observed active preactivations to <.004 and is shared within each difference pair.
- Installed corrected269 (sharp243 plus gate correction) provisionally.
- Requires GPU factual dNLL <= .005; neutral209 is the restore point.

## Forward 25–26; Python 270 — corrected route nearly fits and responds
- corrected269 factual dNLL +.00546 on first 100 tuning stays: narrowly fails .005.
- Dose shifts: insulin x2 -10.857, half +3.275; NE x2 +1.755, half -2.066.
- Original on the same stays: -2.016, +1.051, +.309, -.070.
- Recovering record-based targets and isolating neutral routing footprint before refinement.
- corrected269 remains provisional, not accepted.

## Python 271 — corrected-route refinement audit
- corrected269 remains provisional: factual dNLL +.00546 exceeds the .005 limit.
- Inspecting exact routing changes before reducing their footprint; keeping the empty-action correction.
- No weights changed in this run.

## Python 272 — distinguish response from routing overhead
- Corrected route includes substantial attention-head replacement and FFN compensation.
- Recovering the compensation fit and response columns before tuning; no weights changed.

## Python 273 — alternative-head compensation plan
- Small zero-response route had dNLL +.00509, so routing overhead needs reduction independently of response gain.
- Inspecting reusable attention/FFN features to compare compensated replacement of other heads.
- No weights changed.

## Python 274 — compensation input-contract audit
- Checking actual empty-action vectors, route token returns, and stay-held sample masks.
- Prior compensation predates the empty-action discovery; no revised fit is yet accepted.
- No installed weights changed.

## Python 276 — corrected-input head comparison
- Full stay metadata recovered from stays156.
- Baseline held RMS 0.279580; candidate scores {0: (0.16677933378253695, 0.4549772548072606, 0.0001, 1e-06), 1: (0.21537389484365427, 0.16535948178080018, 0.0001, 1e-06), 2: (0.5716827127207477, 0.41776062063817904, 0.0001, 1e-06), 3: (1.0948381699376113, 1.4025162669999116, 0.001, 1e-05)}.
- Best head 0; provisionally installed True.
- Closed-form regressions only; response directions retained at gain 1.
- No candidate is accepted without GPU factual validation.

## Python 277 — reject alternative-head edit
- Forward 27 alternative-head factual dNLL +0.31001: rejected.
- Restored neutral209; recovering architecture, causal targets, and candidate contracts.

## Python 280 — definitive gain-0.8 installation
- Installed corrected269 with z_in response delta scaled by 0.8.
- Retrieved factual CSV stay IDs directly; candidate remains provisional.

## Python 282 — restart recovery after mixed-type diagnostic failure
- Gain-0.8 corrected269 remains installed and provisional.
- Recovering architecture, recorded causal targets, and latest GPU validation files.
- Explicit NumPy conversion avoids ndarray/Tensor subtraction errors; no weights changed.

## Python 283 — compact validation recovery
- Previous recovery output was truncated before its validation section.
- Extracting recent factual NLLs, dose shifts, exact causal-target notes, and installed response gain.
- No weights changed; gain-0.8 candidate remains provisional pending these results.

## Python 284 — reject gain-0.8 on independent factual cohort
- First cohort dNLL +0.004965; second +0.010455; pooled +0.007685.
- Saved rejected_gain08_284; it is not an acceptable final candidate.
- Installed blended284 provisionally: prelude deltas scaled to 0.6, response gain 1.0.
- Retained the actual-empty-action gate correction. No gradient training.
- Hypothesis: reduce routing overhead while preserving treatment transport; requires GPU validation.

## Python 285 — reject interpolated routing; isolate overhead
- Forward 31 / fwd_32: blended284 dNLL +0.02537 on second 100 tuning stays; rejected.
- Restored full corrected269 route with both added z_in response columns at gain zero.
- zero_route285 is diagnostic only, not accepted. Next factual pass isolates routing overhead.

## Python 286 — reject full routing overhead; correct direct-token test
- Forward 32 / fwd_33 zero-response full route dNLL +0.01055 on second cohort; rejected.
- Installed corrected_direct286: token257 gain 0.3 plus corrected269 empty-action posterior gates.
- All prelude attention and FFN weights restored to original.
- Prior direct-token failure preceded discovery of the nonzero empty-action encoding.
- Hypothesis: fixing that leakage may permit direct causal transport without routing overhead.
- Candidate remains provisional; requires matched factual and dose validation.

## Python 287 — recover direct-token validation context
- corrected_direct286 factual dNLL is +0.00045 on the latest 100-stay cohort.
- This clears the factual limit on that cohort, not yet causal validation.
- Recovering architecture, record-derived targets, saved candidates, and stay IDs.
- No weights changed.

## Python 288 — compact restart recovery
- Prior output was truncated; extracting numeric causal targets, cohort IDs, and edit contracts.
- corrected_direct286 remains installed; factual dNLL +0.00045 on the latest cohort.
- No weights changed.

## Forward 34; Python 289 — direct transport is underpowered
- corrected_direct286 dose shifts on second cohort: insulin x2 -2.743, half +2.309; NE x2 +0.124, half +0.050.
- Factual fit passes (+0.00045), but causal responses remain inadequate.
- Recovering response-channel calibration before changing gains or directions.
- corrected_direct286 retained as a factual-safe checkpoint; no weights changed.

## Python 290 — provisional stronger corrected direct transport
- Installed direct_g1_ne3_290: corrected_direct286 posterior gates, insulin response gain 1.0 and NE response gain 3.0.
- All attention and FFN weights remain original; only the six posterior/input tensors are edited.
- The gain-0.3 checkpoint remains available as a factual-safe fallback.
- Candidate is provisional: requires factual dNLL <= .005 and improved dose responses.
- No gradients or optimiser over model weights used.

## Forwards 35–36; Python 291 — reject higher direct gains
- Gains insulin 1 / NE 3 give factual dNLL +.00214, but dose shifts remain -2.835, +2.120, +.173, +.031.
- Direct transport saturates; larger gains offer no meaningful repair.
- Restored corrected_direct286 as factual-safe checkpoint.
- Investigating corrected-route factual compensation using actual empty-action encoding.

## Python 292 — recover targeted compensation options
- Higher direct-token gains were factually acceptable but causally ineffective; the low-gain checkpoint remains installed.
- Recovering architecture, record-based effect evidence, and available readout/calibration features before spending the remaining runs.
- No weights changed. Any routed candidate must pass independent factual validation and retain its dose responses.

## Python 293 — routed-candidate calibration data collection
- Reinstalled corrected269 provisionally; uncalibrated factual fit is not acceptable.
- Preparing a training-stay state cohort and recovering exact readout helpers.
- Goal: a small record-fitted value calibration, validated on separate tuning stays, without gradient training of the model.
- corrected_direct286 remains the factual-safe restore point.

## Python 295 — provisional protected-head correction
- Run294 completed paired-state and dose-direction collection but stopped at a NumPy/Torch centre mismatch; no correction was installed then.
- Reused collected data: 4415 cases, 4415 recovered labels.
- Held original/routed local NLL: 1.534268 / 1.543273.
- Selected (held dNLL, RMS, local dose distortion, protected rank, ridge): (0.004400521410540925, 0.44859093673445094, 0.02327793711651031, 24, 0.1).
- Refitted local dose shifts: [('insulin', 0.5, 1.287083397155016, 1.3057683364976735, 0.014517271673272289), ('insulin', 2.0, -2.766359706707205, -2.828251869494751, 0.02237314353498017), ('norepinephrine', 0.5, -0.5880330430272229, -0.6095533525665944, 0.036597109285872516), ('norepinephrine', 2.0, 1.0564235511783988, 1.0932019246820195, 0.03481404164323661)].
- Installed protected_head295 provisionally. GPU factual validation on independent tuning cohorts and rollout validation are still required.
- Correction is a closed-form regression, with no model gradients or optimiser over weights.

## Python 297 — final validation recovery
- Run296 failed on a 7-versus-4 intermediate-array mismatch; its diagnostic is inconclusive.
- Recovered architecture, record-derived evidence, checkpoint agreement, and tuning cohorts.
- No weights changed. Protected-head candidate still requires independent GPU factual and rollout validation.

## Python 298 — compact final-validation inventory
- Run297 output was truncated; retrieved checkpoint agreement, tuning cohort IDs, and numeric target notes compactly.
- Preserved the installed tensors in protected_state298. No weights changed.
- Next decision will use matched GPU factual and dose-rollout results, not the failed local diagnostic.

## Python 299 — reject protected-head factual fit; final correction attempt
- Protected-head factual dNLL: second cohort +0.00924; first +0.00373; third +0.00937. Three-cohort pooled deterioration about +0.0074 exceeds the limit.
- Matched second-cohort dose shifts: insulin x2 -10.678, half +4.151; NE x2 +1.471, half -1.733. Causal response survived, but the candidate is rejected as-is.
- Relaxed-head recovery failed safely: IndexError('boolean index did not match indexed array along axis 0; size of axis is 4415 but size of corresponding boolean axis is 4035'). Restored corrected_direct286, previously factual dNLL +0.00045.
- Closed-form statistical regression only; no backpropagation or optimiser over model weights.
- Acceptance still requires GPU factual dNLL <=0.005 and useful dose-response retention. Corrected_direct286 remains the restore point if validation fails.

## Python 300 — final closed-form attempt and bounded safety validation
- Latest safe-checkpoint factual dNLL: +0.00158 nats per reading.
- Fixed twofold application of the previously record-fitted protected readout correction.
- Guard status: awaiting_three_factual_checks.
- No backpropagation or optimiser over model weights used.
- Any background safety work belongs to this run and ends within 850 seconds of its start.
- Proposed weights are provisional until three disjoint paired factual cohorts pass; any failure restores final_safe300.
- Dose-rollout validation is still required; model shifts are not truth.
- Run300 guard fwd_45: factual dNLL +0.02603416, 4664 readings.

## Run300 safety rollback
- fwd_45 factual dNLL +0.02603416 exceeded the +0.0047 safety threshold.
- Restored corrected_direct286; no further edits scheduled.
