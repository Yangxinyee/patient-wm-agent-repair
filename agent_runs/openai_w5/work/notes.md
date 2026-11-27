# World-model repair notebook

## Initial plan
- Infer drug effects only from the supplied simulated-ICU records.
- Inspect architecture, recording order, treatment assignment, and dose support before estimating effects.
- Use non-gradient statistical fits to design targeted weight edits.
- Validate factual value NLL and paired rollout responses against the original model.
- Test uncertainty, confounding, alternative dose levels, start times, and secondary readouts.
- No empirical conclusions yet.

## Architecture inspection (run 1)
- Doses influence posterior latent state and a generic recurrent action pathway.
- Value heads also use a same-code last-reading register and transition logits.
- Evaluation hides future readings but keeps recorded actions, so dose effects must be assessed with multi-step feedback, not just teacher-forced sensitivity.
- No drug-effect estimate yet; inspect assignment variation and measurement timing next.

## Record inspection (runs 2–3)
- 30,000 training stays; 400 tuning stays; approximately 2 million training events.
- Readings occur at integer hours; one-hour infusion records start at hour + 0.0833.
- Insulin rate equals its dose; norepinephrine dose equals 4.8 times its rate.
- Insulin rates are half-unit increments; norepinephrine rates are hundredth increments, capped at 1.
- Constructed hourly panels `panel` and `panel_tune`; absent infusion means zero exposure.
- Need control for treatment assignment, prior readings, patient heterogeneity, and modality selection.
- No edits or forward requests yet.

## Preliminary causal analysis (run 4)
- Simultaneous glucose sensors correlate 0.974; median difference 0.
- Simultaneous MAP sensors correlate 0.900; median arterial minus cuff is -3.
- Built prior-history features and held-out residualized next-hour regressions.
- Nuisances fitted on 12,000 stays; effects estimated on 6,000 different stays.
- These are preliminary conditional estimates, not proof of identification.
- G_next: insulin -1.83442 (SE 0.09964); norepinephrine -0.34592 (SE 0.54635); n=85299.
- M_next: insulin 0.03021 (SE 0.04526); norepinephrine 3.21719 (SE 0.22672); n=122802.

## Original baseline / persistence analysis (forwards 1–3; run 5)
- Tuning A: original value NLL 1.63933; allowable ceiling 1.64433.
- Original A rollout shifts: insulin x2 -0.454, half +0.098; NE x2 +0.150, half -0.232.
- Original states A saved in /work/fwd_3/states_original.npz (4,141 rows).
- Used residualized treatment innovations to estimate future outcome and future dose responses.
- Deconvolved clinician feedback with an AR(1) response model; this requires causal-innovation and stable-linear-response assumptions, to be checked further.
- G dynamic estimates: {'phi': 0.8762725202961656, 'beta': -1.8381467582926028, 'beta_one': -1.834419990988181, 'phi_fixed': 0.8761687969798846}.
- M dynamic estimates: {'phi': 0.6335817172466911, 'beta': 3.5998780750492387, 'beta_one': 3.217188708302707, 'phi_fixed': 0.6682056242010688}.
- Current counts: 5 Python runs, 3 forwards; no edits.

## Sensor checks and correction (runs 8–9)
- Sensor-specific held-out next-hour estimates: glucose_finger insulin -1.8383 (SE .1026), glucose_lab -1.9344 (.1744); map_arterial NE +3.4086 (.2191), map_cuff +2.8671 (.4111).
- Cross-drug coefficients are small relative to their uncertainty.
- Corrected a diagnostic indexing error: clustered_ols includes an intercept. The initial dose-stratum printouts were intercepts, not drug effects, and must be discarded.
- Run 8 dynamic-alternative calculation stopped before completion; no edits were made.

## Local head reconstruction (run 10, forward 4)
- Obtained original states for 100 training stays (4,314 bundles).
- Computing a no-gradient local value-head reconstruction and checking it against original GPU predictions.
- Corrected local-projection dictionary access and calculated ARX/deconvolved alternatives.

## Head reconstruction validation (run 11)
- Simple prior-mean reconstruction had nontrivial error; testing register conventions and latent integration.
- ARX alternatives: glucose phi .7807, beta_now -1.8381, beta_lag -.7382; MAP phi .5116, beta_now 3.2133, beta_lag 1.3739. These remain conditional estimates.

## Restart / run 13
- Run 12's approximate value-head reconstruction retained meaningful mean and NLL errors; no edits are justified by that reconstruction yet.
- Re-reading architecture and inspecting persistent analysis objects before correcting the reconstruction and checking causal estimates.

## Reconstruction conventions / run 16
- Tested schema-valid last-numeric features, last-status embeddings, type embeddings, activations, and prior/head gap scaling.
- Best deterministic convention: {'kind': 'gelu', 'prior_gap': 10.0, 'view_gap': 10.0, 'typeidx': 1, 'status': -1}.
- No tensor edits; sampling and register conventions still require validation.

## Runs 17–18
- Run 17's strata calculation failed because clustered_ols returns more than two objects; corrected by taking its first two outputs.
- Conditional-dose strata: [{'drug': 'insulin', 'stratum': 0, 'n': 276, 'dose_min': -0.060566163015920244, 'dose_max': 0.0008639295968212589, 'beta': 53.65487543780671, 'se': 72.17884213113975}, {'drug': 'insulin', 'stratum': 1, 'n': 56590, 'dose_min': 0.0008639295968212589, 'dose_max': 1.0518725138734473, 'beta': -1.9590038642355183, 'se': 0.22464000466428383}, {'drug': 'insulin', 'stratum': 2, 'n': 28433, 'dose_min': 1.0518725138734473, 'dose_max': 5.871025415817483, 'beta': -1.812524739041011, 'se': 0.11079635884894021}, {'drug': 'norepinephrine', 'stratum': 0, 'n': 1365, 'dose_min': -0.005512806790041913, 'dose_max': 5.095337316459479e-05, 'beta': -108.64595575014621, 'se': 270.3834239022148}, {'drug': 'norepinephrine', 'stratum': 1, 'n': 80486, 'dose_min': 5.095337316459479e-05, 'dose_max': 0.0003305590678662629, 'beta': -38.81679758964102, 'se': 293.3929606869019}, {'drug': 'norepinephrine', 'stratum': 2, 'n': 40951, 'dose_min': 0.0003305590678662629, 'dose_max': 0.9466620452061967, 'beta': 3.210910406717543, 'se': 0.22670434078758578}]
- Insulin has broad observed rate support (0–16); norepinephrine observed rates are capped at 1. Above-cap effects need a response-form assumption.
- Best bin/flag trial: (0.013937794603407383, 0.24990640580654144, 'left', 'minutes', '3').
- No edits. Dynamic rollout targets remain assumption-dependent, not observed truth.

## Register search / run 19
- Selected reconstruction convention: {'binside': 'left', 'layout': (6, 3, (0,)), 'ageunit': 'seconds'}.
- Insulin slopes agree across informative strata; near-zero-dose strata have inadequate innovation variance and do not identify effects.
- Provisional AR(1) tuning-A targets: insulin double -11.893, half +5.946; NE double +4.497, half -2.249. These are estimated, not truth.
- No weight edits.

## Restart after forward 5
- Independent training states received in /work/fwd_5.
- Rechecking architecture and reusable analysis objects before selecting an edit.
- No edits; causal targets remain estimates conditional on identification and response-form assumptions.

## Run 23: head-only correction screening
- Testing the selected register convention on original GPU predictions.
- Preparing independent state samples for a no-gradient correction of treatment innovations.
- Head-only edits would preserve the recurrent state and all other prediction heads directly.

## Run 25: posterior-action ablation candidate
- Head-only innovation decoding did not generalize (held-out R2 approximately zero).
- Latent sampling accounts for much of the local-vs-GPU head discrepancy.
- Exploratory edit: zero z_post.0.weight[:,384:576], retaining direct action input.
- Hypothesis: action-conditioned latent severity inference masks direct drug responses.
- Requires paired factual and rollout tests; this is not yet a validated repair.
- Earlier small-sample error-vs-innovation SE printouts include an intercept SE.

## Forward 6 / run 26: failed posterior-action ablation
- Removing z_post.0.weight[:,384:576] barely changed any dose-edit response.
- Reverted the ablation; no tensors currently edited.
- This does not support posterior action conditioning as the main cause of weak responses.
- Reinspecting direct dose/state pathways before choosing another intervention.

## Runs 27–33: local encoder reconstruction
- Confirmed state layout: out = core.out_norm(mean of four h slots).
- Drug-start latent reconstruction uses GELU, quality embedding row 0, and encoded empty measurement set.
- Insulin: dose unit 3, rate unit 2; NE: dose unit 4, rate unit 3.
- Recorded decimal doses resolve NE bin-boundary discrepancies; left-sided binning confirmed.
- Measurement/empty-action and infusion-END conventions are being checked before any recurrence edit.
- No tensors edited; 6 forward passes used.

## Restart, runs 42–43
- Re-read architecture. Doses enter recurrent state; value heads also use sensor-specific last-reading registers.
- Run 42 inventory failed on a module's callable shape attribute, before any edit.
- Best local recurrence remains approximate; current edits are empty, 6 forwards used.

## Runs 45–48: dose representation and failed decoder
- Dose encoder varies sharply and non-monotonically with rate; infusion END has no dose attributes and goes through the measurement encoder.
- Ridge decoding of treatment innovations from posterior hidden features, including synthetic dose multipliers, failed held-out factual and dose-response tests. Do not use it.
- Temporary diagnostic: action-encoder norm output fixed to the verified empty-action vector. Test paired factual NLL before deciding whether replacement is feasible.

## Restart / forward 7
- Empty-action ablation: factual NLL 1.65737 versus original 1.63933, +0.01804; fails the +0.005 constraint. Reverted all edits.

## Restart, runs 55–59
- Re-read architecture and inspected local recurrence.
- Run 58 stopped on context tuple/tensor mismatch; no edit.
- Run 59 tests dose/innovation decoding at drug bundles on stay-held-out splits.

## Restart after run 61
- Run 61 found two local latent-response correction directions, but no weights were edited.
- Complete encoder RMSE remained about 0.084; recurrence diagnostics included NaNs and large errors.
- Local mean response matching is not sufficient evidence of a valid rollout repair.
- Reinspect architecture and persistent candidate objects before testing any edit.

## Runs 65–66: reconstruction bookkeeping bug
- Forward states are in interleaved stay order, not sorted by stay ID.
- np.searchsorted on stay IDs selected incorrect memory boundaries; fixed local diagnostic boundaries.
- Zeroing empty-action vectors worsened posterior reconstruction and was not retained.
- Held-out action-u linear rate decoding active R2: insulin .653, NE .519; innovation R2 .281/.180.
- No weight edits. Revalidate recurrence conventions using saved posterior means.

## Runs 67–68: provisional response reconstruction
- Global attention packing and current-inclusive memory improved local recurrence substantially.
- Best local convention remains approximate; it cannot certify factual fit or rollout effects.
- Recomputed latent response directions across stay-held-out samples and checked finite-sized shifts.
- No weights edited. Any eventual candidate requires paired GPU factual and rollout validation.

## Restart after run 68
- No edits retained. Infinitesimal latent-response fits did not yield stable finite-innovation responses.
- Re-reading architecture and inspecting persistent objects before selecting a finite-dose, GPU-validated candidate.
- Small-perturbation agreement alone is not evidence of causal rollout accuracy.

## Runs 74–75: response randomness audit
- Run 74 stopped on time-gap broadcasting; no edits or conclusions.
- Run 75 corrects the gap shape and repeats the identical-state test.

## Restart after run 75
- Identical-state test exposed randomness in the old readout; the corrected readout was exactly repeatable.
- No edits retained. Rechecking architecture and corrected routines before drawing finite-dose conclusions or submitting an edit.

## Run 80: corrected latent-response audit
- With shared random draws, posterior-latent Jacobian norms were roughly 0.0001–0.0009 reading units per latent unit.
- Earlier large latent-response directions were sampling artifacts and are rejected. No edits.
- Next examine the direct action/recurrent pathway.

## Runs 84–88: action-path screening
- Re-read architecture; no edits retained.
- Run 83's counterfactual decoder introduced excessive factual variation and is rejected.
- Checking the direct action/recurrent response with fixed states and common random numbers.

## Run 95: finite action-direction screening
No edits. Tested the run-91 action injection directions with common random numbers.
(3, 'glucose_finger', 0, [-0.6821, -0.6821, -0.6823, 0.0, -0.7587, -1.086, -1.1593, -1.1255, -1.0838])
(1732, 'glucose_finger', 0, [0.8659, 0.8659, 0.8639, 0.0, -0.1293, -0.1191, -0.127, -0.1328, -0.1321])
(2941, 'glucose_finger', 0, [2.5046, 2.5045, 2.4997, 0.0, 0.3413, 1.0609, 1.5254, 1.7237, 1.8131])
(57, 'map_arterial', 1, [-0.7036, -0.7027, -0.679, 0.0, -0.5635, -0.6827, -0.6927, -0.6972, -0.6994])
(981, 'map_arterial', 1, [-0.3143, -0.3679, -0.4208, 0.0, -0.1269, -0.2168, -0.2157, -0.2144, -0.2137])
(2037, 'map_arterial', 1, [-1.5759, -1.6021, -1.5988, 0.0, -0.2656, -1.4317, -1.474, -1.4724, -1.4708])

Run 95 rejects the large action directions: finite effects saturate and frequently reverse sign. No weights edited. Moving to a GPU-validated, targeted value-head correction.

## Runs 97–99: information audit
Head hidden features fail independent insulin innovation decoding; with-code NE decoding reaches only R2=.072. A simple head-only repair is unsupported. Inspecting earlier prelude information before constructing any isolated causal channel. No edits.

## Restart after run 99
- No edits retained. Prelude innovation decoding has held-out R2 about .20, but finite-dose correction fits caused excessive factual variation.
- Reinspect exact architecture and persistent helpers before choosing an isolated dose pathway. All candidates require finite-dose and paired GPU factual validation.

## Run 103: diagnostic state-coordinate clearing
- Reserved coordinates [61, 21, 15, 66] by zeroing their c_out and coda residual-output rows.
- This is only a capacity diagnostic, not a causal repair. No dose signal has been installed.
- Original rollout responses are near zero despite record-derived insulin/NE effects.
- Next: paired factual test; reject this route if clearing alone consumes the +.005 NLL budget.

## Forward 8 / run 105: private-channel feasibility
- Clearing four state coordinates costs only +.00034 factual value NLL on tuning A.
- Prepared, but did not install, frozen-feature rate regressions: {'alpha': 1e-05, 'rmse_I_NE': [0.40887, 0.13455], 'absent_rmse_I_NE': [0.043429, 0.024621], 'max_abs_I_NE': [10.9567, 2.4564]}.
- Fixed input representation saturation limits: [11.52, 3.0].
- This rate-encoding fit uses no model gradients and makes no new causal-effect claim.

## Run 110: private-channel decoder audit
- Audited the final-layer decoder by treatment configuration.
- Tested direct-atom and first-hidden rate/identity recovery on independent continuous doses.
- No additional weight edits; encoding accuracy alone is not causal or factual validation.

## Run 116: isolated-channel capacity test
- Refit rate and drug-presence decoding on original first-hidden action features; no gradients.
- Temporarily zeroed one prelude attention-head output (minimum output-weight norm), in addition to the four cleared state coordinates.
- This is an ablation diagnostic, not a causal repair. Paired factual testing determines whether its capacity cost is acceptable.

## Forward 9 / run 117
- Freeing prelude head 3 costs +0.00246 factual value NLL on tuning A (1.64179 versus 1.63933).
- Prepared an analytic first-hidden rate channel and linear encoder distillation; not installed.
- Encoder validation report: {'alpha': 0.001, 'rate_rmse_I_NE': [0.136104, 0.056676], 'rate_max_error_I_NE': [4.05774, 0.82129], 'ordinary_u_rmse': 4.050224742968567e-05, 'absent_rate_rmse': [0.0, 0.0]}

## Run 118: normalized analytic rate channel
- Tested exact GELU-pair rate arithmetic with normalized-output distillation to avoid q times unknown layer-normalization scale.
- Best local report: {'alpha': 0.001, 'rate_rmse_I_NE': [0.004566, 0.001215], 'typical_rate_rmse_I_NE': [0.003383, 0.000982], 'max_rate_error_I_NE': [0.09879, 0.03036], 'ordinary_u_rmse': 0.0021985468920320272, 'raw_sigma_min_max': '0.98269 / 1.00664'}
- No encoder candidate installed yet.

## Run 123: encoder candidate installed for factual validation
- Independent rate RMSE I/NE: [0.004455155805703784, 0.0014733773277843163].
- Ordinary action-feature RMSE: 0.0027597.
- Installed the best normalized analytic encoder; no causal readout coupling yet.
- Existing coordinate/head ablations remain. GPU factual validation is required.

## Run 125: isolated dose-transport candidate
- Routed head 3 to the action token using a carrier discriminator and exact constant query.
- Private state coordinates receive paired rate signals and decay before subsequent updates.
- No causal value-head correction installed. This is a transport/capacity test only.
- Must pass GPU factual NLL and saved-state rate-transport checks before use.

## Restart after run 125
- Transport routing passed the local action-token selection check only.
- No causal readout coupling is installed; factual and temporal transport validation remain required before retaining this candidate.

## Forward 11–12 / transport validation
- Tuning A paired factual value NLL: original 1.63933, transport candidate 1.64142, increase 0.00209.
- This passes the constraint but leaves only 0.00291 for the causal readout correction.
- Paired states saved in /work/fwd_13; dose recovery and temporal behavior are being audited before coupling.

## Runs 130–134: restart and head audit
- Re-read architecture and earlier record-derived effects. Dynamic targets remain conditional on identification and response-form assumptions.
- Run 133 stopped on a layout-dictionary/tuple mismatch; no weights changed.
- Run 134 rebuilt reading features using explicitly edited weights and aligned readings with preceding-hour exposures. No causal head edit yet.

## Run 135: clean-pair capacity candidate
- Equalized paired coordinates in out/query normalization and additive head inputs; removed incidental old value-head pair sensitivity.
- Amplified private dose transport by 400. This is not a causal calibration.
- Compensated both atom encoders by linear algebra to preserve primitive embedding preactivations. No gradients or optimizer used.
- Candidate requires paired GPU factual validation; retain only if the +.005 value-NLL constraint holds with room for a causal correction.

## Forward 15 / run 138: clean-channel factual validation
- Clean-pair candidate value NLL on tuning A: 1.64123 versus original 1.63933; +0.00190.
- Passes the +0.005 constraint on this sample; approximately 0.00310 budget remains.
- Saved-state ordinary out RMS difference is 0.03842. Pair differences are small and require timing/rate auditing.
- No causal value-head coupling installed yet; factual fit alone is not a causal validation.

## Restart after run 141
- Four-neuron head distillation has negligible local likelihood cost; no causal head edit was installed in run 141.
- Dose-proxy error and heterogeneous local tilt sensitivity remain unresolved.
- Recovering architecture, record-derived identification results, and persistent calibration objects before another edit.

- Run 145 inspection failed on a module's shape attribute; no weights changed.

## Runs 147–149: sensor-specific causal-head candidate
- Run 147's zero insulin tilt was a constrained-fit sign bug, corrected in run 148; it is not evidence of an absent insulin effect.
- Freed eight low-energy value-head neurons by frozen-feature ridge distillation; no model gradients.
- Separate, clean sensor gates; smooth monotone logit tilts calibrated to record-derived one-step effects.
- Centered decoded exposure using independent-stay-validated ordinary-feature regressions.
- Installed insulin strength 1 and NE strength .5 to reserve factual-loss budget.
- Float32 local holdout NLL increment 0.001359 above clean transport; GPU factual and finite-dose tests still required.

## Forward 16–17: sensor-specific candidate GPU validation
- Tuning A factual value NLL: original 1.63933, edited 1.63946, delta +0.00013.
- Paired finite rollouts: insulin x2 -5.591, half +0.901; NE x2 +4.083, half -1.052.
- Record-derived tuning-A estimates: -11.893, +5.9465, +4.4973, -2.2487.
- Correct directions, but strong doubling/halving asymmetry. Increasing strengths alone is not yet justified.
- Investigating exposure transport, centering, nonlinear head response, and hourly feedback.

## Runs 153–155: asymmetry diagnosis
- Intended sensor control gates are approximately 8: GELU gate clipping is ruled out on saved factual features.
- Fixed-dose perturbations affect only the intended reserved neuron; ordinary hidden neurons are unchanged.
- Held-out dose recovery RMSE: finger .144, lab .135, arterial .116, cuff .072.
- Testing finite softmax response separately from transport and recurrent feedback; no new weights edited.

## Runs 156–161: restart / finite-transport audit
- Fixed-history ideal-dose decoder response is approximately linear and symmetric.
- GPU hourly asymmetry is already visible at hour 13; finite action transport must be checked before attributing it solely to feedback.
- No additional weight changes. Auditing helper weight references and exact current-encoder routing.

## Run 178: CPU/GPU recurrence audit
- Shared helper weights and duplicate CPU implementations agree exactly.
- Both still disagree with saved GPU states; local transport conclusions remain unvalidated.
- Re-reading architecture and recovering diagnostic objects before further edits.

## Run 182: checkpoint alignment
- Found s4 matched ORIGINAL states in fwd_15, whereas replay weights were EDITED.
- Earlier CPU/GPU discrepancies therefore do not establish a recurrence or transport defect.
- Switched s4 to fwd_15 edited states and repeated boundary/memory checks; no weights changed.

## Runs 185–189: replay recovery
- Re-read architecture and record-derived estimates after restart.
- Refreshed the stale world-encoder normalization cache.
- Private recurrent coordinates nearly match edited GPU states; ordinary replay still does not.
- Auditing finite-dose private transport with temporary record copies; no new weight edits.

## Run 191: diagnostic correction
- Run 189's apparent routing discontinuities used a stale world cache; they are not evidence of a GPU routing defect.
- Installed candidate_e184 into the local replay cache only. No model weights changed.

## Restart / run 193
- Preserved all current edits in _r193_checkpoint; no weights changed.
- Recovering architecture, causal estimates, and replay dependencies.
- END atoms and stale replay caches remain diagnostic concerns, not established causes.

## Run 202: clean recovery after timeout
- Kernel variables were lost in run 199; persisted edits survived.
- Saved all current edits in /work/checkpoint_run202.safetensors.
- Rebuilt weight caches directly from the current and original tensors; no weights changed.

## Restart after forward 18 / run 206
- Tuning B factual value NLL: 1.62038 versus original 1.61898, +0.00140.
- Factual preservation holds on this sample; finite-dose causal asymmetry remains unresolved.
- Recovering architecture, record-identification evidence, and kernel objects before further edits.

## Runs 209–211: recovered finite-rate channels
- Correct clean pairs are insulin (21,61), NE (15,66); initial diagnostic pairing was wrong.
- Encoder signals are mostly linear and drug-specific. Insulin saturates near rate 11.49 due to input clipping.
- Saturation alone cannot explain doubling response exceeding twice the halving response.
- Next hypothesis: ordinary-feature exposure centering may absorb dose interventions; not yet established.

## Run 217: restart recovery and rollback snapshot
- Saved current edits to /work/checkpoint_run217.safetensors; no weights changed.
- Recovered sensor-specific identification checks and current head/centering objects.
- Existing exposure centering uses ordinary features; intervention leakage remains a hypothesis, not a finding.

## Runs 218–219: local-head validation failure
- No weights changed. Local head NLL (~2.85) disagrees with GPU factual NLL (~1.64).
- Local strength-loss and centering diagnostics are not trusted until register/head reconstruction is validated.

## Runs 226–229: reconstruction audit
- Re-read the architecture and record-identification notes; no weight edits.
- Exact weight caches do not resolve local/GPU prediction disagreement.
- Auditing prior sampling, register timing, and state alignment before using local likelihood calculations.

## Runs 230–232: concrete diagnostic corrections
- Correct register layout is consistent with [presence, age, z_value, has_value, z_dose, has_dose, z_rate, has_rate]; the recovered layout was wrong.
- Recomputed outcome and last-value bins from each sensor's supplied edges.
- No weight edits. Validating prior-averaged paired predictions before further calibration.

## Run 233: bin-boundary audit
- Prior averaging did not validate the recovered local head.
- The register-layout improvement is partial, not proof of exact reconstruction.
- Testing edge equality explicitly; changed bin indices must not automatically be treated as corrections.
- No weight edits.

## Run 234: restart and validation policy
- Left-edge binning agrees with existing labels; boundary changes do not explain local/GPU disagreement.
- Recovered CPU likelihood calculations remain unvalidated and will not determine edit acceptance.
- Recovering identification evidence and edit structure; subsequent acceptance requires paired GPU factual and rollout tests.

## Run 238: GPU-calibrated strength trial
- Saved complete pre-trial edits in /work/checkpoint_run238.safetensors.
- Multiplied paired glucose value-head columns by 2.0 and MAP columns by 1.2; all transport and centering weights unchanged.
- This is a calibration trial, not an accepted repair. Requires paired GPU dose rollouts and factual NLL validation.
- Scaling alone is refuted if doubling/halving asymmetry persists or factual value-NLL cost exceeds +0.005.

## Forward 22 / run 243: reject strength-only trial
- Tuning A factual NLL increase +0.00536 exceeds the +0.005 ceiling.
- Restored the pre-trial value-head output weights (glucose strength 1, NE baseline).
- Investigating whether post-dose ordinary features leak dose into the centering regression.

## Run 249: reversible centering-isolation trial
- Saved all pre-trial edits in /work/checkpoint_run249.safetensors.
- Replaced ordinary-feature exposure centers with sensor-specific constant factual exposure means; retained private-dose coefficients and value-head output strengths unchanged.
- This is a diagnostic, not an accepted repair. GPU paired rollouts must establish whether removing center leakage reduces asymmetry; factual NLL must be tested before retention.

## Forward 23 / run 250: constant-center diagnostic rejected
- Constant-center rollout shifts: insulin x2 -4.152, half +1.090; NE x2 +4.538, half -1.092 (reached stay-hour means).
- Removing ordinary-feature centering did not resolve asymmetry. Restored the pre-trial first-layer value-head tensors.
- Next audit: exact action encoding and attention transport for WORLD, END, and simultaneous start/end bundles.

## Run 251: bundle-semantics audit
- END and WORLD+END synthetic signals are irregular; these are not yet validated explanations of rollout asymmetry.
- No weights changed. Re-reading architecture and recovering exact action/transport helpers before another trial.

## Runs 252–257: persistence audit
- Stronger glucose readout alone gives little rollout gain and costs factual NLL; retained baseline strengths.
- Factual reconstructed signal neurons are near +8, not GELU-clipped.
- Private core.flow.a=0 resets dose coordinates within seconds. Multi-hour persistence therefore relies on ordinary autoregressive feedback.
- Comparing hourly GPU effects with record-derived AR response before a reversible persistent-channel trial; estimates remain assumption-dependent.

## Run 258: persistence-trial preparation
- Hourly paired effects understate record-derived insulin responses, especially halving.
- Rapid private-channel decay is a candidate mechanism, not an established explanation.
- Recovering exact architecture, causal estimates, and current edit layout before a reversible trial.
- No weights changed.

## Runs 261–263: finite transport audit
- Current insulin WORLD encoder is essentially linear through rate 8; saturation starts near 11.5.
- Plausible END variants add an offset but do not reproduce the large doubling/halving asymmetry.
- Synthetic END semantics remain unverified; no conclusion rests on them alone.
- Completed NE audit and recovered prior rollout summaries; no weights changed.

## Run 264: reversible insulin-persistence trial
- Saved all pre-trial edits in /work/checkpoint_run264.safetensors; BASE264 holds all tensors.
- Insulin coordinates 21/61 now decay with tau=1500 seconds; core.a_in pair equalized to preserve antisymmetry.
- NE transport and all value-head weights unchanged.
- This is not accepted. Requires GPU factual, state stability, and paired finite-rollout checks.
- Variable normalization makes retention approximate; factual centering may need recalibration.

## Forward 24–25 / runs 265–266: persistence trial rejected as-is
- Insulin tau=1500 s: factual NLL 1.64604 vs 1.63933 (+0.00671), exceeds guardrail.
- Rollouts: insulin x2 -8.110, half +1.952; NE x2 +4.362, half -1.332.
- Magnitudes increased, asymmetry remains. Trial saved in checkpoint_trial264.safetensors.
- Restored pre-trial flow; dropped a_in edit because checkpoint omitted this previously original tensor.

## Run 270: prepare factual-centering recalibration
- Reinstalled only trial flow/a_in from checkpoint_trial264; still an unaccepted diagnostic.
- Plan: fresh training GPU states, cross-validated ordinary-feature centering, then GPU factual and rollout acceptance.

## Run 275: persistence-centering trial
- Matched fresh tau=1500 s training states to pre-trial training states.
- Fitted only the incremental insulin factual offset, with five-fold stay-group CV.
- Added ordinary-feature corrections to glucose control neurons; private drug coefficients unchanged.
- Saved checkpoint_before_center275.safetensors. Trial remains unaccepted pending GPU factual and rollout tests.

## Forward 27 / run 276: recentering alone insufficient
- Tuning A factual NLL 1.64550 vs 1.63933, +0.00617: fails guardrail.
- Candidate remains unaccepted. Inspecting signed factual errors before either intercept calibration or shorter persistence.

## Restart recovery, runs 277–281
- Recovered architecture, provisional record-derived targets, and private-channel layout.
- Current tau=1500 recentered candidate still fails tuning-A NLL by +0.00617.
- Glucose-finger accounts for most deterioration despite improved aggregate mean; an intercept-only repair is not supported by this diagnostic.
- No weights changed during recovery; candidate remains unaccepted.

## Run 282: reversible value-distribution calibration
- Saved all pre-calibration edits in checkpoint_run282.safetensors.
- Scaled all three value-logit components by 0.97, including transition logits.
- Drug transport, persistence, and first-layer centering unchanged.
- Motivation: glucose-finger NLL worsens despite improved aggregate mean; distributional calibration is a hypothesis, not an established cause.
- Requires paired GPU factual validation and subsequent finite rollouts; not an accepted candidate.

## Forward 29 / run 283: temperature trial insufficient
- Scaling all value logits by 0.97 gives tuning-A NLL 1.64527 vs 1.63933, +0.00594: still fails.
- No weights changed in this recovery run. Candidate remains unaccepted.
- Recovering architecture, record-derived targets, and saved edit objects before a smaller reversible adjustment.

## Run 285: reversible shorter-persistence trial
- Saved the failed tau=1500/temperature candidate in checkpoint_run285.safetensors.
- Undid the 0.97 temperature scaling. Set insulin private coordinates 21/61 to tau=900 seconds.
- Retained 25% of the incremental tau=1500 factual-centering correction; baseline centers and drug coefficients unchanged.
- NE and other transport unchanged. This is unaccepted pending paired GPU factual and rollout checks.

## Forwards 30–31 / run 286: feasible baseline and intermediate trial
- Tau=900 tuning-A factual NLL 1.63892 vs original 1.63933: -0.00041.
- Tau=900 paired shifts: insulin x2 -5.633, half +1.384; NE x2 +4.235, half -1.083.
- Saved this locally feasible candidate in checkpoint_feasible900.safetensors.
- Trial: insulin tau=1250 s, with 60% of the incremental tau=1500 factual center. All strengths unchanged.
- Trial is unaccepted pending factual and rollout checks. Record-derived target estimates remain assumption-dependent.

## Run 287: restart recovery and acceptance audit
- Tau=1250 rollout: insulin x2 -6.728, half +1.549; NE x2 +4.374, half -0.953 on tuning A.
- Recovering its paired factual score and earlier record-derived targets before retaining or changing it. No weights changed.

## Run 288: compact recovery
- Earlier conditional estimates support insulin beta about -1.83 and NE beta about +3.2 to +3.6; no medical priors used.
- Provisional tuning-A AR targets: insulin -11.893/+5.946, NE +4.497/-2.249. These are not observed intervention truth.
- Retrieving latest factual scores and exact edit controls; no weights changed.

## Run 289: reversible glucose-strength trial
- Tau=1250 baseline factual A: 1.639911 vs 1.639331, delta +0.000580; passes locally.
- Saved all pre-trial edits in checkpoint_run289.safetensors.
- Multiplied only the four glucose signal/control output columns by 1.40. Transport, persistence, centers, and NE unchanged.
- Candidate is unaccepted pending paired factual and rollout checks, including fresh tuning stays. No gradient training.

## Forward 34 / run 290: reject glucose-strength 1.40
- Fresh tuning B factual NLL: 1.62730 vs 1.61898, delta +0.00832. Fails the local guardrail.
- Reverted only the value-head output weights to BASE289 (tau=1250, glucose strength=1.0).
- Next paired factual test on the same fresh stays distinguishes strength cost from persistence/centering generalization failure.

## Run 291: recover after duplicate-ID request failure
- Failed forward request provided no evidence and was not counted.
- Recovering unique fresh-test stays and the current tau=1250 baseline.
- No weights changed; fresh-stay paired factual validation remains required.

## Run 292: compact fresh-stay acceptance audit
- Inspected paired factual results in forwards 35 and 36 and checked current weights against checkpoint_run289.
- No weights changed; decisions will use the paired losses, not the failed duplicate-ID request.

## Run 293: complete 400-stay factual audit
- Tau=1250 pooled factual NLL delta: 0.00414485 nats per reading.
- Stay-bootstrap 5/95% limits: [0.002395111462150231, 0.005861597770799757].
- Groups C and D individually exceed +0.005; pooled margin is small.
- No weights changed. Inspecting centering/calibration before choosing a safer final candidate; no further strength increase is justified yet.

## Run 294: sensor-calibration preparation
- Current factual glucose predictions have an upward shift, but a mean correction alone is not yet known to repair their NLL.
- Inspecting reserved signal/control neurons and per-code fine-bin geometry for a small, non-gradient intercept calibration.
- No weights changed. Any calibration requires paired factual and finite-rollout validation.

## Run 295: reversible glucose-intercept calibration trial
- Saved pre-trial weights in checkpoint_run295.safetensors.
- Used odd-ID tuning stays for conservative moment-based glucose offsets; even-ID stays are the calibration validation subset.
- Trial offsets: {"glucose_finger": {"signal_neuron": 74, "delta_bias": 0.030854555095138802, "calibration_mean_bias": 0.9042828841951932, "validation_mean_bias_before": 0.8528624870107376, "approximate_mean_shift": -0.7234263073561547, "output_slope": -0.11030532131572154, "slope_fit_rmse": 0.6816574213880858}, "glucose_lab": {"signal_neuron": 107, "delta_bias": 0.01567748534628584, "calibration_mean_bias": 0.5364056660039763, "validation_mean_bias_before": 0.6015071225071228, "approximate_mean_shift": -0.42912453280318114, "output_slope": -0.16931620147095142, "slope_fit_rmse": 0.9919856942639208}}
- No gradients or weight optimizer used. Drug effect estimates remain those from training records, not these calibration residuals.
- This candidate is unaccepted pending paired factual NLL and finite-rollout checks; tau=1250 and all drug coefficients unchanged.

## Run 296: restart recovery and calibration audit
- Latest calibrated factual group: 1.62788 vs 1.62352, delta +0.00436.
- Recovered architecture, earlier record analysis, paired factual files, rollout files, and validation-group IDs. No weights changed.
- Calibration remains provisional until pooled factual validation and paired finite-rollout checks; training-record causal estimates are not observed intervention truth.

## Run 297: compact paired calibration audit
- Computed pre/post-calibration paired losses on matching stay groups, including even-ID validation subsets and stay-bootstrap uncertainty.
- Recovered exact group IDs and latest rollout schemas. No weights changed.
- Missing post-calibration groups must be evaluated before claiming full 400-stay factual feasibility.

## Run 298: complete matched calibration audit and nonlinear-head inspection
- Exact pre-calibration 400-stay value-NLL delta: 0.00414485.
- Exact calibrated delta: 0.00434379; calibration loss change: 0.00107603.
- Intercept calibration reverted: True. Moment correction is not retained when it worsens the actual NLL.
- Saved both snapshots; current accepted-baseline snapshot is /work/checkpoint_baseline298.safetensors.
- Inspected candidate signal/control pairs: {74: 171, 107: 7}. A shared-offset test is only justified if output columns cancel and code gating remains intact; no shared-offset edit made yet.
- The empirical pooled guardrail is not a guarantee on unseen stays. Total loss over non-value outputs and unavailable intervention rollouts remain unverified.

## Run 299: final reversible persistence/centering trial
- Rejected shared-offset idea: earlier reconstructed factual signal units are near +8, not demonstrably GELU-clipped.
- Installed insulin private-coordinate tau=1100 seconds, with factual centering interpolated 0.57142857 from the saved 900-second candidate toward the validated 1250-second baseline.
- Preserved private drug coefficients, output-head strengths, NE transport, and every unrelated tensor.
- Trial saved in /work/checkpoint_trial299.safetensors; rollback snapshot is /work/checkpoint_baseline298.safetensors.
- Acceptance requires full 400-stay paired factual testing and paired dose rollouts. Seek lower loss than +0.00414485 without losing substantial dose responses; otherwise restore the exact baseline.
- Python runs: 299/300. Forward passes: 42/50. One Python run is reserved for final audit, notebook update, and conditional rollback.

## Run 300: final provenance audit and exact ID recovery
- The attempted numeric tuning-ID range was invalid; that request was not counted
  and supplied no evidence about the model.
- Recovered 400 tuning IDs corroborated by successful factual files,
  partitioned into 4 batches of at most 100.
- Retained candidate: 1100-second insulin persistence/centering trial.
- Trial mismatches: [].
- Changes from baseline: [('attr_heads.value_bin.0.bias', 0.008510589599609375), ('attr_heads.value_bin.0.weight', 0.05572628974914551), ('core.flow.a', 0.12783336639404297)].
- Unexpected changes: [].
- No gradient training or weight optimization was performed.
- Python budget is exhausted: 300/300. Forward budget remains 42/50.
- Weights are now frozen. The remaining forward passes must audit the retained
  candidate on paired factual predictions and finite dose rollouts.
- For the 1100-second trial, the prior 400-stay baseline delta of +0.00414485
  is NOT its measured loss. Full factual acceptance remains unverified.
- Record-derived drug-effect estimates remain assumption-dependent, not observed
  intervention truth. Total non-value loss and unsupported interventions remain
  unverified.
