# World-model repair notebook
Python runs: 1. Forward requests: 0.
Goal: establish dose effects from the simulated records, then edit weights without gradients while preserving factual value NLL within +0.005 nats.
No medical-effect assumptions. No edits made yet.
First step: read architecture, data schema, vocabulary, and tensor shapes.

Python runs: 3; forward requests: 0. Loaded 1,998,495 training rows (30,000 stays), 400 tuning stays. Measurements are hourly; drug records start at hour+0.0833, last one hour. Insulin rates in 0.5 increments; norepinephrine rates in 0.01 increments, capped at 1.0. Run 2 tensor summary failed because load returns torch tensors; fixed with as_numpy. Constructed hourly arrays A/AT, times -1..24.

Python runs: 4; forward requests: 1. Original rollout on tuning IDs 2020000000..99: insulin x2=-0.422, half=+0.213 glucose units; NE x2=-0.742, half=-0.238 MAP units. Created Xhist (recorded pre-dose history), stay-separated twofold histogram-regression residual estimates of immediate dose effects. Results: {"('insulin', 'glucose_finger')": {'beta': -1.9117976582589598, 'se': 0.04643702514945556}, "('norepinephrine', 'map_arterial')": {'beta': 3.2284108103034193, 'se': 0.09160659418894486}}

Python runs: 5; forwards: 1. Stronger 350-iteration, twofold history-adjusted estimates on all four sensors; also residualized dose-squared to check nonlinearity. {"('insulin', 'glucose_finger')": {'beta': -1.93686465753238, 'se': 0.045618852592945285, 'quadratic': [-1.9721614199729816, 0.006519443110720444], 'quadratic_se': [0.10001575970918698, 0.01835970138120731]}, "('insulin', 'glucose_lab')": {'beta': -1.9077086863023527, 'se': 0.07546108816200746, 'quadratic': [-2.1349505733834957, 0.043098195029487404], 'quadratic_se': [0.178045837967501, 0.035021676061520214]}, "('norepinephrine', 'map_arterial')": {'beta': 3.1919950557110304, 'se': 0.09112374089621184, 'quadratic': [1.6021787887032015, 1.2276868502178129], 'quadratic_se': [0.30275496930303175, 0.22235442673524064]}, "('norepinephrine', 'map_cuff')": {'beta': 2.6932493742117205, 'se': 0.17140609404668736, 'quadratic': [-1.6426030664176126, 3.4223180731807608], 'quadratic_se': [0.572236102544827, 0.4254051172466354]}}

Python runs: 6; forward requests: 1. Robust immediate estimates: insulin finger -1.9369, lab -1.9077; little quadratic evidence. NE arterial +3.192, paired pre-dose sensors +3.329, high expected rates +3.407; cuff +2.693 with confounding near treatment onset. Estimated controlled impulse kernels by sequentially removing later dose contributions, using cross-fitted baseline treatment innovations. Kernels: {"('insulin', 'glucose_finger')": [-1.917764, -2.287755, -1.986765, -1.581741, -1.23612, -0.906992, -0.6303, -0.489408, -0.329701, -0.199948, -0.288551, -0.267519], "('insulin', 'glucose_lab')": [-1.972058, -2.386697, -2.017123, -1.797053, -1.249801, -0.946148, -0.647348, -0.623698, -0.385423, -0.467155, -0.216104, 0.05464], "('norepinephrine', 'map_arterial')": [3.195033, 2.710725, 1.587524, 1.030973, 0.377504, 0.218539, 0.359975, -0.138514, 0.138292, -0.039861, 0.393475, -0.090408]}

Python runs: 9; forwards: 2. Recovered architecture and variables. Original factual value NLL on first 100 tuning stays is 1.63982. No weights edited. Preparing record-derived rollout targets.

Python runs: 10; forwards: 3. Original training states acquired (100 stays, 4313 bundles). Record-derived reference shifts matched to original rollout reading rows: {"insulin_x2": -12.95553151488211, "insulin_half": 6.477765757441055, "ne_x2": 4.279678475624698, "ne_half": -2.139839237812349}.

Python runs: 12; forwards: 3. Matched single-start z posterior locally to ~1e-10 MSE. Action units: insulin unit=3/rate_unit=2; NE unit=4/rate_unit=3. MLP activation GELU; observation empty bundle is encoded (not zero). Full posterior END reconstruction: [(0.02114708721637726, 'preserve'), (0.010521634481847286, 'clear_numeric'), (0.012452274560928345, 'clear_all')]. No edits yet.

Python runs: 13; forwards: 3. Restart recovery: inspecting architecture and live variables. No weights edited; full-stream posterior reconstruction is not yet exact.

Python runs: 17; forwards: 3. Isolated drug starts match posterior to ~1e-10 MSE. Testing measurement units/kinds and END observation-vs-action routing. Results: {'glucose_finger': [(0.013916670344769955, (1, 4, 1)), (0.014012562111020088, (1, 4, 0)), (0.01431475579738617, (0, 4, 1))], 'glucose_lab': [(0.01767573133111, (5, 4, 1)), (0.018761184066534042, (5, 4, 0)), (0.022612350061535835, (5, 4, 2))], 'map_arterial': [(0.00836174376308918, (2, 3, 0)), (0.00865178368985653, (2, 1, 2)), (0.008657408878207207, (2, 3, 2))], 'map_cuff': [(0.004897202365100384, (2, 4, 2)), (0.00496144313365221, (2, 4, 0)), (0.004988657310605049, (2, 3, 2))], 'insulin END': [(0.006666292902082205, ('clear_all', 4, 4, 3, 'action')), (0.006666292902082205, ('clear_numeric', 4, 4, 3, 'action')), (0.006941919680684805, ('clear_numeric', 3, 4, 3, 'action')), (0.006941921543329954, ('clear_all', 3, 4, 3, 'action'))], 'norepinephrine END': [(0.007430164143443108, ('clear_all', 1, 2, 3, 'observation')), (0.007430167403072119, ('clear_numeric', 1, 2, 3, 'observation')), (0.0074330768547952175, ('clear_all', 2, 4, 3, 'observation')), (0.007433077786117792, ('clear_numeric', 2, 4, 3, 'observation'))]}

Python run 21; forwards 3. Restart recovery: no edits. Prior measurement/END posterior reconstructions remain inaccurate. Reading full architecture, schema, tensor list and live helpers before further encoding tests or weight edits.

Python run 22; forwards 3. Architecture recovered; previous output truncated before live helpers. Inspecting compact metadata and helpers. No edits.

Python run 25; forwards 3. Clean context checks and measurement metadata enumeration (unit, kq, bin equality, optional categorical fields); no edits.

Python run 26; forwards 3. Metadata-only explanation rejected. Testing flow/input scaling and END conventions; no weights changed.

Python run 27; forwards 3. No exact match from metadata, flow/input scale or END searches. Verifying cached weights, state layout and numeric ordering before choosing an edit mechanism. No submitted edits.

Python runs 29-30; forwards 3. Numeric bins use strict edges below x. Run 29 metadata printed; global inspection failed on a module's shape function. Safely recovering live arrays/helpers. No edits.

Python run 31; forwards 3. Cached original states and causal reference targets remain available. Narrow inspection of helper definitions and observation diagnostics. No edits.

Python run 33; forwards 3. Corrected an inspection helper that did not represent empty sets. Testing WORLD/REVEAL/END kind explicitly with clean single-atom encoding. No edits.

Python run 34; forwards 3. Drug-start posterior reconstruction remains exact, but observations do not. Recovering architecture and live helper definitions to isolate the observation-path discrepancy before any weight edit.

Python run 35; forwards 3. Architecture confirms separate action and observation DeepSets encoders. Run 34 output was truncated before helpers; requesting bounded metadata and helper index. No edits.

Python run 36; forwards 3. Checking set cardinality, bundle alignment, helper bytecode and original-weight cache consistency. Empty-observation mismatch makes preprocessing/cardinality a priority. No edits.

Python run 38; forwards 3. Run 37 failed only on return-shape printing before comparisons. Repeating full-bundle and cardinality-stratified posterior tests with safe return handling. No edits.

Python run 39; forwards 3. Correct bundle cardinality did not resolve observation errors. Testing required zero action vector for empty action bundles, while drug-start matches remain exact. No edits.

Python run 40; forwards 3. Observation encoder remains unresolved; exact drug-start path and authoritative posterior states are available. Testing stay-held-out linear extraction of history-adjusted dose innovations from the 32 saved posterior means, with 20x weight on inactive bundles. Also fitting posterior reading-response directions conditional on the original state context. Candidate repair is a low-rank posterior-mean edit, subject to factual and rollout validation. No edits yet.

Python run 41; forwards 3. Linear extraction of dose innovations from posterior z failed (held-out R2 approximately zero). Fitting direct action-vector rate/presence extractors, augmented at x0/.5/1/1.5/2/3 and three END conventions, plus pre-dose-context expected-rate regressions. Proposed gated GELU-pair correction is zero without a WORLD drug atom and proportional to rate minus contextual expected rate when present. No edits yet.

Python run 42; forwards 3. Restart: action-rate extractors are not accurate enough for a safe edit. Inspecting architecture and live helpers to choose an exact dose-sensitive correction path. No weights edited.

Python run 45; forwards 3. Posterior MLP is 576->192->64; no edits. Estimated sensor-specific latent and state writing directions conditional on authoritative preceding context, to guide a causal channel rather than use medical assumptions.

Python run 47; forwards 3. Conditional augmented action-rate fits still have RMSE .642 insulin/.319 NE; insulin presence classifier FP .147. Run 46 ended on a float64/float32 diagnostic mismatch, now fixed. Checking actual factual action bundles and recovering the existing expected-dose target fits; no edit submitted.

Python run 49; forwards 3. Run 48 resolved the extractor scaling: X is the physical action vector, and earlier fitted coefficients require X/scale. On actual factual action bundles, active-dose RMSE is .321 insulin/.189 NE, with near-zero inactive presence. Recovering architecture and retained helpers before a bounded correction test. No weights edited.

Python run 50; forwards 3. Architecture reread. Recovering compact helper/fit metadata and original dose-rollout scale; no edits.

Python run 51; forwards 3. Assessing gated posterior-mean repair using physical action decoding and context-predicted dose. Original response is much weaker than record-derived insulin effect and has the wrong doubled-NE sign. No edits yet.

Python run 52; forwards 3. Refit fresh dose/presence regressions directly on verified physical action vectors after another retained-coefficient scaling inconsistency. Checking policy predictions and latent directions before any edit; no weights changed.

Python run 53; forwards 3. First pilot submitted: four low-importance posterior hidden units replaced by two gated GELU differences. Each writes a glucose/MAP latent direction proportional to decoded dose minus contextual expected dose, inactive without a WORLD drug. Other posterior outputs compensated by closed-form hidden-feature regression. Pilot strengths -2 insulin/+3.3 NE, from record-derived immediate effects. Four z_post tensors edited; factual and rollout validation still required.

Python run 55; forwards 5. Run 54 had a parse-time SyntaxError, so the four-tensor pilot from run 53 remains installed and unvalidated. Recovering architecture, repair helpers, and validation stay IDs.

Python run 56; forwards 5. Architecture reread. Inspecting existing forward 4/5 results before requesting more passes; run 55 output was truncated, not an execution failure.

Python run 58; forwards 5. Run 57 stopped at Tensor.copy before any modification. Pilot factual NLL difference +2.13e-8. Reached-row shifts original -> pilot: insulin x2 -0.422 -> -0.406, half +0.213 -> +0.199; NE x2 -0.742 -> -0.726, half -0.238 -> -0.230. Pilot remains ineffective. Saved pilot_weights_53 as NumPy arrays; inspecting scaling/gating.

Python run 60; forwards 5. Run 59 was a parse-time failure; no changes. Repeating posterior-gate and latent-path diagnostics.

Python run 61; forwards 6. Authoritative original/edited tuning states acquired. Comparing actual posterior and recurrent-state changes and auditing local posterior reconstruction against training states.

Python runs 63-64; forwards 6. All pilot edits reverted. z_in has ordinary weight scale, so failure cannot be attributed simply to a tiny latent projection. Recovering retained state data and exact core layout.

Python run 65; forwards 6. Original model restored. Authoritative states and action encodings retained; inspecting prelude attention and later repair fits to identify an effective causal injection path.

Python run 66; forwards 6. Replayed original prelude on authoritative states to quantify token attention and normalized-token covariance. No edits.

Diagnostic probe: exact GELU pairs expose selected context, observation and action coordinates in posterior means. Temporary four-tensor edit; restore immediately after acquiring probe states. Prelude latent attention on drug bundles is <1e-5, explaining the ineffective posterior pilot.

Python run 68; forwards 7. Probe states acquired for 100 training stays. All four diagnostic posterior edits immediately dropped; original model restored. Recovering architecture, statistical evidence and live helpers before choosing an effective correction path.

Python run 69; forwards 7. Original model restored and architecture recovered. The direct action-token/recurrent pathway is the candidate target; recovering the record-derived causal effects and retained design matrices before calibration.

Python run 70; forwards 7. Recovered evidence: insulin immediate effect about -1.94/unit with persistent kernel; NE about +3.2/rate with shorter persistence. Original tuning rollout reference targets: insulin x2 -12.96, half +6.48; NE x2 +4.28, half -2.14. Inspecting retained state/prelude helpers; no edits.

Python run 71; forwards 7. Comparing exact probe posterior input coordinates against retained local encodings and context reconstruction, with original weights restored. This diagnoses preprocessing rather than attempting another ineffective latent edit.

Python run 72; forwards 7. Probe mean columns expose 0.05 times selected inputs; scaling was explicitly diagnosed and corrected before interpreting comparisons. Testing context and encoder agreement stratified by observations, drug starts and END bundles. No edits.

Python run 73; forwards 7. Scale correction alone did not resolve probe comparison. Auditing affine offsets and exact probe tensor construction before trusting its input coordinates. No edits.

Python run 74; forwards 7. Identified retained probe output bias. Exact exposed inputs are (saved z_mean - r67_probe_bias)/0.05; context residual after removing its affine offset was approximately 1e-8. Comparing correctly decoded e/u by bundle type, including normalized-empty action baseline. No edits.

Python run 75; forwards 7. Original weights remain restored. Probe establishes exact drug-start encodings and reveals nonzero empty-action encoding. Recovering update equations and retained helpers to reconstruct the effective action/recurrent pathway; posterior-only correction was ineffective because its attention was negligible.

Python run 76; forwards 7. Architecture confirms effective action-token prelude/recurrent path. Inspecting its exact operations and retained record-derived response fits before designing a dose-innovation edit. No weights edited.

Python run 77; forwards 7. Prelude FFN modifies slots immediately before their normalization and recurrent update: a potentially effective dose-innovation insertion point. Retrieving compact identification evidence and retained matrices after run 76 output truncation. No edits.

Python run 78; forwards 7. Record-derived kernels suggest persistent insulin lowering and shorter positive NE response; original rollout responses are far smaller. Preparing exact drug-bundle prelude features to test extraction of dose innovations after attention. No edits.

Python run 79; forwards 7. Empty e/u encoder baselines now match the probe exactly. Isolating measurement-unit and overlapping END/WORLD routing conventions before prelude dose-innovation fitting. No edits.

Python run 80; forwards 7. Tested measurement units and six END routing/attribute conventions against exact probe coordinates. Best units {'glucose_finger': 0, 'glucose_lab': 5, 'map_arterial': 2, 'map_cuff': 2}; best convention (4.15297772232248e-14, 'clear_all', 'observation', 3.268759435760653e-14, 8.842182865618272e-15). No weight edits.

Python run 81; forwards 7. Exact preprocessing recovered: glucose_finger unit=0, glucose_lab=5, MAP=2; END atoms route to observations with cleared numeric/unit fields; empty action is its encoded baseline. Verified full-stream posterior and reconstructed prelude features. No weight edits.

Python run 82; forwards 7. Full posterior RMSE 2.42e-5; action attention ~0.232 versus latent ~3e-7 on WORLD drug bundles. Reconstructing recurrent update; best convention (0.0020365272648632526, 1.4436134733841754e-05, True, 3600.0, 1.0, 0). No weights edited.

Python run 83; forwards 7. Restart recovery: the seconds/current-state update reproduces authoritative out with RMSE 1.44e-5 and h with RMSE .00204. Recovering equations/helpers to target the effective prelude/recurrent action path, with all weights original.

Python run 84; forwards 7. Architecture confirms a non-negligible action-token prelude path. Recovering exact replay functions and record-only identification evidence before constructing a factual-policy-residual correction. No edits.

Python runs 86-87; forwards 7. Run 86 found fn76 returns selected action bundles, not the full stream; its indexing diagnostic failed without changing weights. Built an explicitly indexed WORLD-drug action re-encoder, excluding END markers, to validate prelude dose contrasts. No edits.

Python run 88; forwards 7. Exact WORLD-drug re-encoding has zero RMSE. Built exact prelude cache and fitted two shared-slot FFN-hidden dose contrasts, testing x0/.5/1.5/2/3 and holding out whole stays. Factual covariance penalty is intended to keep the edit small on recorded inputs; no weights edited yet.

Python run 89; forwards 7. Initial hidden-dose contrast accuracy was insufficient. Tested lower ridge penalties and direct per-head attention-pooled normalized-token features, with held-out stay validation and factual-score variance checks. No weights changed.

Python run 90; forwards 7. Attention dose sensitivity is concentrated in head 2. Comparing held-out average dose contrasts across factors and inactive factual scores; preparing actual preceding-reading registers for a finite-difference readout probe. No edits.

Python run 91; forwards 7. Original weights remain restored. Recovering exact architecture and recent helpers to replace weak existing-feature dose extraction with a direct, factual-residual correction.

Python run 94; forwards 7. No edits. Recovering compact causal summary and state-update interface after diagnostic output/name failures.

Python run 95; forwards 7. Recovered record-derived immediate and impulse effects (insulin ~-1.94/unit, NE ~+3.2/rate); tuning rollout references -12.96/+6.48 and +4.28/-2.14. Preparing mechanistic finite-difference readout calibration, no gradients.

Python run 96; forwards 7. No edits. Matching local readout reconstruction to existing authoritative factual outputs before calibrating core injection.

Python run 97; forwards 7. Brute-force matching readout conventions against authoritative original factual means, using prior mean and known registers. No weights changed.

Python runs 99-100; forwards 7. Original weights remain restored. Architecture re-read; recovering dose-extraction fits, identification evidence, and exact readout conventions before an effective recurrent edit.

Python runs 103-104; forwards 7. Latent-token gate is strongly isolated: ordinary-token SD .00092, changed-action SD .00064 versus latent mean 1. Testing explicit dose decoder and low-variance attention-payload features for a posterior innovation channel. No weights edited yet.

Python run 105; forwards 7. Refined explicit dose decoder and calibrated head-2 value-channel directions with numerical finite differences; exact prelude/core replay checked. No model gradients or weight edits.

Python run 107; forwards 7. Calibration is not yet trusted: local readout and core disagreement require exact replay diagnosis. Original weights remain unchanged; recovering compact helper inventory.

Python run 112; forwards 7. Dose-policy residual noise assessed (insulin .60, NE .25 on active bundles). New readout helper disagrees with earlier reconstruction; checking authoritative recurrent states and memory timing before any weight edit.

Python run 118; forwards 7. Original weights unchanged. Recovered replay helpers; explicitly tested seconds versus hours for memory age against authoritative drug-bundle states.

Python run 122; forwards 7. Context/view replay is close, but readout/register reconstruction remains unvalidated. Recovering architecture and retained helper inventory before calibration. No weights edited.

Python run 128; forwards 7. Correct query type is 1. Prior-mean readout RMSE 1.766, averaged-prior RMSE 1.677, four-draw noise SD 1.156. Local readout is approximate; factual acceptance must use authoritative forwards. Moving to a reversible dose-contrast pilot rather than further exact-readout search.

Python run 130; forwards 7. Restart recovered architecture and early identification evidence. No edits. Original rollout is severely attenuated relative to record-derived insulin and NE kernels; proceeding to a reversible action-path pilot with authoritative factual validation, not further exact-readout reconstruction.

Python run 133; forwards 7. Retained Bdelta88 and payload104 fail direct paired-contrast validation (large error versus near-zero response); neither applied. Refit attention-pooled dose contrasts directly with whole-stay holdout and factual-covariance regularization. Original weights remain unchanged.

Python run 136; forwards 7. Fresh FFN contrast fit has low factual variance and consistent insulin signs but misses NE stopping/halving; unsuitable as final full-dose solution. Constructed finite-difference low-rank injection directions and x2 contrast normalization for a reversible pathway pilot. No weight edits yet.

Python runs 137-141; forwards 7. Recovered architecture and identification evidence. Record-derived immediate effects ~-1.94 insulin/glucose and +3.2 NE/MAP; 12-hour tuning reference shifts -12.96,+6.48,+4.28,-2.14. Candidate FFN hidden decoder has small factual residuals in each slot (insulin SD ~.06, NE ~.034); no cancellation across slots. Installed reversible rank-two correction to core.prelude.0.ffn.2 weight and bias: dW=directions136@B136.T, db=-dW@mu136, gains [1,1]. Directions are numerical finite-difference calibrations, no model gradients. Factual acceptance and actual rollout effects remain untested. NE halving/stopping decoder is known imperfect; this is a pathway pilot, not a final edit.

Python run 142; forwards 9. Rank-two FFN pilot rejected and dropped. At gain [1,1], factual NLL rose 1.63982->1.64743 (+.00761), exceeding budget. Actual rollout shifts were -0.570,+0.194,-0.721,-0.303, versus originals -0.422,+0.213,-0.742,-0.238. Thus local finite-difference calibration did not translate into meaningful rollout repair. All weights restored. Investigating an explicit latent/action dose-innovation transport channel.

Python run 143; forwards 9. Original weights remain restored. Recovering architecture, data schema, and earlier record-only causal evidence after failed FFN pilot; next correction must validate dose transport through actual rollout, not only local readout sensitivity.

Python run 144; forwards 9. Record-derived insulin and NE effects disagree strongly with original rollouts. Recovering exact action/prelude/core helpers and explicit dose decoder to validate transport before another edit. No weights changed.

Python run 145; forwards 9. Testing a dedicated latent-transport attention head, with zero payload first. Replayed original prelude and fitted three-head compensation on factual training states with whole-stay holdout; no weights changed.

Python run 146; forwards 9. New prelude replay failed equality checks, so no head edit is justified. Comparing retained exact helper and raw-versus-normalized token conventions; original weights remain unchanged.

Python run 147; forwards 9. Raw-token replay validated. Dedicated latent head 3; three-head and closed-form FFN compensation, held-out E RMSE 0.000725809. Zero-payload candidate installed=True; factual acceptance still requires authoritative forward. No dose payload yet.

Python run 148; forwards 10. Zero-payload latent-head transport passed tuning factual test (1.63982 original, 1.63985 edited). Checking retained nonlinear action decoder and context dose-policy accuracy before adding a causal signal.

Python run 149; forwards 10. Restart recovery: zero-payload latent-head transport remains installed and factual-neutral (+0.00003 NLL). Retained action decoder rate RMSE is about 0.021 insulin and 0.016 NE; context-policy prediction errors are much larger. Recovering exact transport, decoder, and record-identification evidence before adding any dose payload. No weights changed in this run.

Python run 150; forwards 10. Architecture recovered. Current five-tensor edit reserves an attention transport head with factual compensation but no causal payload. Recovering its compact implementation and retained nonlinear decoder before a payload contrast test. No weights changed.

Python runs 151-152; forwards 10. Run 151 diagnostic stopped on structured reference data; no weight changes. Recovered reference structure, prior posterior-pilot notes, decoder scenario checks, and reserved-head projection norms.

Python run 153; forwards 10. Reconstructed original and saved gated-posterior pilot on aligned factual and scaled action cases, measuring latent contrasts before transport. Previous pilot rollout failure must be distinguished from decoder failure. Current five zero-payload transport edits remain unchanged; saved posterior pilot was not installed.

Python run 154; forwards 10. Constructed, but did not install, four-neuron gated physical-dose innovations with fresh whole-stay-held-out context policies, compensated posterior outputs, and two explicit orthogonal latent carrier directions. Candidate dictionaries post154/zin154/zb154 retained. Transport and physiological output calibration still required; current five zero-payload edits unchanged.

Python run 157; forwards 10. Posterior replay is authoritative (RMS 2.4e-5), but recurrent replay and readout calibration are not validated. Recovering architecture and retained transport/posterior candidates before further edits. Existing five-tensor zero-payload transport remains unchanged.

Python run 162; forwards 10. Recurrent replay explicitly tested learned core.age embeddings, key-only versus source insertion, seconds/hours, log scaling, and memory inclusion. Best configuration: (0.002028598915785551, 1.4811970686423592e-05, True, 3600.0, 1.0, 0). No weights changed.


Python runs 165-169; forwards 10. Recovered architecture, record-derived effects, and state helpers. No additional edits; saved zero169 reversible transport checkpoint. Checking raw-seconds learned age keys, current-bundle memory, and latent payload units before a numerically calibrated transport pilot.

Python run 172; forwards 10. Run 171 calibration was ill-conditioned and failed output-space reconstruction; no additional weights installed. Recovering architecture and retained objects before validating a unit payload transport and recalibrating physiology. Five zero-payload transport tensors remain edited.

Python run 173; forwards 10. Architecture and early record-only estimates recovered. Run 172 output truncated before retained-object inventory; requesting compact recent inventory. Existing zero-payload transport edits unchanged.

Python run 174; forwards 10. Inspecting reserved attention-head value/output layout and calibration restrictions. Hypothesis: changing the reserved output block can avoid the ill-conditioned original 48-direction span. No new weights installed.

Python run 175; forwards 10. J171 is visibly float32-quantized at epsilon .02, so its inverse directions are not trusted. Inspecting readout alignment, head payload, and layer structure before larger-perturbation calibration. Current edits unchanged.

Python run 176; forwards 10. Reserved head value matrix is rank one, so its unused value channels permit unrestricted output directions. Checking factual dose-signal error and locating recurrent helper before recalibration. No additional edits.

Python runs 177-179; forwards 10. Runs 177-178 stopped on type mismatches, without edits; candidate posterior and latent replay nevertheless align exactly. Standardized NumPy diagnostic arrays and tested actual-carrier dose decoding across held-out stays and six scales. Current zero-payload edits unchanged.

Python run 180; forwards 10. Run 179 shows factual-trained carrier decoding substantially attenuates physical-dose contrasts, especially insulin, and fails uniform scaling/stopping. No new edits. Inspecting exact posterior/transport implementation before a held-out multi-scale contrast calibration.

Python run 181; forwards 10. Architecture confirms dose transport passes nonlinear posterior/prelude/recurrence. Run 180 inventory truncated; retrieving selective helper signatures and record-identification notes. No weights changed.

Python run 182; forwards 10. Recovered early record-derived impulse kernels and tuning references; substantial effects are evidence-based, not clinical assumptions. Inspecting posterior and transport equations to isolate contrast attenuation. Existing zero-payload edits unchanged.

Python run 184; forwards 10. Old J171 is dominated by float32 quantization and is not a trusted physiology calibration. Testing readout sensitivity independently before any further weight edit. Five zero-payload transport edits remain unchanged.

Python runs 185-186; forwards 10. Recurrent replay RMS 0.002185; large raw-state norms explain tiny small-epsilon derivatives. Run 185 stopped only at a NumPy/Torch diagnostic mismatch, without edits. Run 186 computes an unrestricted 192-direction central-difference calibration at epsilon 8 and checks nonlinear sensor responses; all weight edits remain the factual-neutral zero-payload checkpoint.

Python run 187; forwards 10. Four-sensor calibration is nonlinear/ill-conditioned for NE; prioritize glucose_finger and map_arterial, retain alternate-sensor diagnostics. Refit full normalized-latent linear carriers on paired physical-dose contrasts at 0/.5/1.5/2/3 with factual covariance penalties and whole-stay holdout. No new edits yet.

Python run 188; forwards 10. Restart recovery: low-penalty latent decoding preserves insulin contrasts but NE stopping/high scales remain attenuated, and factual residuals are not negligible. Recovering architecture, original record-based identification, and retained replay/calibration objects before installing a payload. Existing five zero-payload transport edits unchanged.

Python run 189; forwards 10. Architecture and schema recovered. Prior record-only impulse estimates show substantial insulin and NE effects; original rollout is near-zero/wrong-sign. Run 188 output truncated before helper inventory, so retrieving compact targets and retained objects. No weight changes.

Python run 190; forwards 10. Recovered record-derived reference means: insulin x2 -12.9555, half +6.4778; NE x2 +4.2797, half -2.1398 on first 100 tuning stays. Inspecting exact latent carrier and unrestricted prelude-response helpers before a reversible actual-rollout pilot. No weight changes.

Python run 191; forwards 10. Exact carrier uses edited posterior and normalized z_in output. Retrieving decoder layout, constant attention channel, and prelude compensation outputs to build one reversible transport pilot. No weight changes.

Python run 192; forwards 10. Built compensated latent transport with low-penalty paired-contrast dose decoder and primary physiological directions. No-payload whole-stay-held-out E RMS=0.00072583323; pilot installed=True. This is a reversible pilot, not validated repair; factual NLL and actual rollouts must decide acceptance.

Python run 193; forwards 12. Pilot hash 3161239253 factual NLL 1.64438 vs 1.63982 (+.00456). Actual rollout shifts: insulin x2 -15.255, half +1.669; NE x2 +2.075, half -1.991. First meaningful transport success, but dose asymmetry and alternate-sensor errors need correction. Saved checkpoint193; investigating primary-preserving multi-sensor directions and temporal rollout shape. Current pilot unchanged.

Python run 194; forwards 12. Restart recovery: pilot gives meaningful but asymmetric effects, with factual NLL +0.00456 near the limit. Recovering exact retained decoder and transport before distinguishing carrier error from downstream rollout feedback. No weights changed.

Python run 195; forwards 12. Architecture recovered. Inspecting paired normalized-latent contrasts and exact calibration helpers to distinguish bidirectional dose-decoding error from nonlinear transport. No weights changed.

Python run 196; forwards 12. Checking actual held-out paired latent contrasts and prelude compensation before tuning signal gains. Current pilot remains installed; no weights changed.

Diagnostic after run 196; forwards 12. Previous output repeated run 196. Testing baseline drift and symmetric local transport; no weights changed.

Python run 198; forwards 12. Baseline pilot glucose rollout drifts +4.55 units. Local insulin transport is symmetric, while NE transport saturates and crosses into glucose. Tracing scaled action signals through actual attention and FFN, and inspecting exact carrier isolation. No weights changed.

Python run 199; forwards 12. Actual prelude insulin contrasts remain bidirectional. Testing an exact two-carrier-only decoder that cannot respond to original latent features, with explicit zero on no-drug bundles, before a reversible rollout pilot. No weights changed.

Python runs 200-207; forwards 12. Recovered pilot and exact posterior. Both full and restricted decoders transported a huge insulin signal on U=0 (about -168 to -170 dose units); attention did NOT suppress it. Disabling the four added gate neurons removes this to <1e-6. Action-normalization constant is 1 within 4e-6 on all recorded drug bundles, 0 when U=0. Shifted two insulin gate biases into this action-only constant and replaced full latent decoder by two-carrier restricted199. Run 206 stopped before saving due to a tensor-type mismatch, fixed in 207. candidate206 installed=False; reversible checkpoint193 retained. Next: authoritative factual and common-random-number rollout tests.


Python run 208; forwards 12. Restart recovery: candidate206 suppresses no-action signals, but is not installed. Inspecting architecture and retained candidate/checkpoint/install helpers before authoritative factual and rollout validation. No weights changed.


Python run 209; forwards 12. Verified candidate206 tensor shapes and finiteness, retained reversible r209_snapshot, and installed the restricted-carrier/action-only gate candidate. Next tests use the same 100 stays as pilot193. Acceptance requires factual NLL increase <=0.005 and improved common-random-number dose effects.


Python run 210; forwards 14. Gate-fixed restricted candidate hash b0e94cf0b5: factual NLL 1.64302 vs 1.63982 (+.00320). Rollout shifts insulin x2 -19.175, half +1.421; NE x2 +1.701, half -1.245, versus record-derived references -12.9555,+6.4778,+4.2797,-2.1398. No-action leak fix alone does not resolve insulin asymmetry. Inspecting exact decoder and hourly dynamics. Candidate remains installed; checkpoint193 retained.


Python run 211; forwards 14. Hourly insulin half shift is -1.796 at hour 13 and remains weak throughout, whereas x2 grows to -26.18 by hour 24. Investigating asymmetric carrier or dose-following factual compensation rather than assuming one scalar gain can repair both. No weights changed.


Python run 212; forwards 14. Run 211's decoder matrix output truncated the useful inventory. Retrieving compact paired-dose inputs and helper returns to test actual signal contrasts. No weights changed.


Python run 213; forwards 14. Testing candidate206 normalized latent decoder against physical dose changes on paired action embeddings with fixed context, including stopping and tripling. Also checking rollout baseline drift. No weights changed.


Python run 214; forwards 14. Fixed-context carrier insulin slopes are 1.004 (stop),1.063 (half),.976 (double),.983 (triple); NE slopes .70,.825,.687,.663. Insulin carrier alone does not explain rollout asymmetry. Baseline rollout drift vs original is glucose +9.08 and MAP -3.28. Testing attention and FFN compensation contrasts plus local replay. No edits.

Python run 215; forwards 14. Insulin carrier and attention contrasts are bidirectional; rollout asymmetry appears downstream, not a simple dose-decoding failure. Recovering recurrence and transport helpers before testing generated-history feedback. No weights changed.

Python run 216; forwards 14. Architecture confirms all drug effects traverse latent/prelude/recurrence; recovering retained replay computations and record-derived physiological targets for a symmetric-impulse test. Current candidate206 remains unchanged.

Python run 217; forwards 14. Testing installed physiological directions with equal signed fixed-history impulses, isolating downstream nonlinearity from generated-history feedback. No weights changed.

Python run 218; forwards 14. Fixed-context raw insulin impulses are approximately symmetric at ordinary amplitudes; NE is saturating and cross-affects glucose. Testing complete changed-dose token/prelude/recurrent responses, including the pre-existing action pathway. No weights changed.

Python run 219; forwards 14. Previous complete-token replay was skipped due to ambiguous tuple extraction; corrected to prelude return[0], as used by preludeE185. Comparing complete candidate dose changes to its own factual baseline. No weights changed.

Python run 220; forwards 14. Complete one-update dose responses have the expected signs; asymmetry emerges during longer rollout. Refitted factual compensation to current restricted-carrier/action-only-gate features, with paired-dose contrast penalties and whole-stay holdout. Candidates retained but NOT installed. Scores: [(0.0, 0.0320853665471077, 2.3988113915265274, 17.64769950264696), (1.0, 0.12415843456983566, 8.088883218800747, 4.642133222522818), (10.0, 0.17595066130161285, 11.477848959018463, 0.9316270506559936), (100.0, 0.18675681948661804, 12.230851381727947, 0.15495143162362993)]

Python run 221; forwards 14. Generated baseline is not near a glucose ceiling. Installed reversible current-feature compensation refit with paired-dose contrast penalty 1; checkpoint221 retains the prior candidate206. Edit hash 333459b508ac. Next: authoritative factual test on the same tuning stays; accept only <=0.005 nats increase before rollout testing.

Forwards 15-16; Python run 222 recovery. Refit factual NLL 1.64091 vs 1.63982 (+0.00109), but rollouts -18.901,+0.915,+1.601,-1.255 versus record-derived -12.9555,+6.4778,+4.2797,-2.1398. Factual improvement retained as checkpoint222; rollout asymmetry unresolved. Recovering exact implementation before further edits.

Python run 229; forwards 16. Restart recovery: run 228 shows correct baseline arrays but large, nonlinear full-sensor transport directions. Recovering architecture and retained implementation before any further weight changes. Current checkpoint222 edits unchanged.

Python run 230; forwards 16. Architecture confirms recurrent action transport and generated-reading register feedback. Inspecting replay helpers and record-derived causal dynamics to isolate the source of asymmetric rollout effects. No edits changed.

Python run 231; forwards 16. Recovering the exact FFN insertion and compensation decomposition. Working hypothesis: nonlinear/off-manifold transport or generated-history-dependent compensation causes dose asymmetry; identification remains based only on records. No weights changed.

Python run 232; forwards 16. Checking whether transport directions exceed ordinary state scale and whether local readout calibration includes actual sensor registers. No edits changed.

Python run 234; forwards 16. Run 233 ended on NumPy/Torch subtraction; no edit changes. Recovered attention decomposition and tested after-FFN transport at next-reading delay by finite differences, without gradients or weight optimization. Candidates are diagnostics only.

Python run 235; forwards 16. Found sensor-order bug in T4_228: correct order is cuff, finger, arterial, lab. Full-sensor diagnostic trials were mis-targeted; current primary repair and record estimates unchanged. Computed correctly ordered raw directions and isolated actual paired-dose attention transport. No weights changed.

Python run 236; forwards 16. Restart recovery: corrected sensor order does not explain the substantial mismatch between intended and actual paired-dose transport. Recovering architecture, identification notes, and exact installed implementation before making further edits. No weights changed.

Python run 237; forwards 16. Recovered record-derived main rollout references from early notebook. Retrieving compact implementation inventory after run 236 output truncation. No weights changed.

Python run 238; forwards 16. Inspecting replay helpers, installed direction scales, and FFN edit structure to diagnose bidirectional rollout asymmetry. No weights changed.

Python run 239; forwards 16. Actual transport aligns with installed dd193 basis; large edits and nonlinear recurrence/compensation remain suspects. Recovering payload decomposition for reversible gain/regularized-direction experiments. No weights changed.

Python run 240; forwards 16. Verifying cached fitting data against installed checkpoint222 before constructing reversible payload-gain tests. No weights changed.

Python run 241; forwards 16. Found cached w2_220/b2_220 are original weights; installed factual raw RMS differs. Verifying fixed attention-feature contrasts for a small treatment-innovation decoder and rank-two FFN adjustment. No weights changed.

Python run 242; forwards 16. Extracting physical doses directly from retained record rows and fitting a held-out treatment-contrast decoder on fixed FFN features. No weights changed.

Python run 243; forwards 16. Corrected physical-dose extraction to exclude END records and verified against rate88. Fitted balanced 0/.5/2/3 treatment-contrast decoder with whole-stay holdout. Installed reversible rank-two FFN candidate243 (insulin gain increment -.35, NE +1.0); checkpoint243 retains validated checkpoint222. Next: authoritative factual and paired-rollout validation.

Python run 244; forwards 18. Candidate243 failed factual budget: +0.00595 nats. Rollouts -19.054,+1.339,+1.817,-1.064 scarcely changed despite intended rank-two gain adjustment. Verified checkpoint243 restoration: True. Recovering exact architecture and helpers; simple gain interpretation is not supported.

Python run 245; forwards 18. Restored validated checkpoint222 via checkpoint243. Inspecting transport decomposition and normalization scales after rank-two adjustment had little rollout effect. No new weight changes.

Python run 246; forwards 18. Bounded transport/helper inventory after output truncation. Record identification supports approximately linear insulin dose effects and positive NE effects. Checkpoint222 remains installed; no new edits.

Python run 247; forwards 18. Repair directions have norms 261 and 940 and prelude changes greatly exceed original scales. Inspecting exact payload/compensation implementation before a whole-payload scaling test. No new edits.

Python run 248; forwards 18. Attention-output delta matches the large payload scale; candidate243 altered only FFN. Retrieving no-payload checkpoint and fitting data to scale attention payload itself, with refitted factual compensation. No new edits.

Python run 250; forwards 18. Run249 safety check rejected an incomplete dd193 decomposition without modifying weights. Recovered complete carrier-associated directions; relative residual 5.65e-16. Installed candidate250 from checkpoint193 with actual insulin payload .25 and NE 1.5, unmatched component unchanged. Checkpoint249 retains checkpoint222 for rollback. Awaiting authoritative factual and rollout tests.

Python run 251; forwards 18. Restart recovery: candidate250 remains installed and unvalidated. Read architecture/data documentation and recovered prior factual-validation stay IDs before authoritative tests. No weights changed.

Python run 252; forwards 20. Candidate250 rejected: factual +0.00818 nats; rollouts (-20.393,-1.319,+2.027,-2.226). Insulin response does not follow attention-payload gain. Restored validated checkpoint222 from checkpoint249. Inspecting joint transport/compensation before further edits.

Python run 253; forwards 20. Checkpoint222 remains restored. Inspecting exact latent, attention, and FFN helpers plus calibration cache to diagnose coupled compensation. No weights changed.

Python run 254; forwards 20. Implemented direct weight-based prelude decomposition and checked fixed features. Preparing total raw-state factual/dose-contrast regression rather than separate attention gain scaling. No weights changed.

Python run 255; forwards 20. Verified factual dose/latent token construction and recalibrated total raw-state directions by central finite differences around original factual states. No model gradients or weight changes.

Python run 256; forwards 20. Restart recovery: local total-state calibration is nonlinear and has sensor crosstalk. Recovering architecture, record identification, and retained implementation before attempting a jointly fitted factual/dose-contrast correction. No weights changed.

Python run 257; forwards 20. Run256 output truncated before implementation inventory. Recovered original rollout and record-derived references; requesting compact helper/cache inventory. No weights changed.

Python run 258; forwards 20. Current FFN compensation penalized only its own dose contrasts, not the total edited prelude response. Inspecting decomposition and cached changed-dose tokens for a total-response constrained regression. No weights changed.

Python run 259; forwards 20. Verifying factual and multi-scale changed-dose cache alignment and prelude return ordering before jointly constraining the total raw-state response. No weights changed.

Python run 260; forwards 20. Fresh factual reconstruction differs from cached prelude features. Installed raw activations are large; checking normalization and replay scales before choosing total-response regression targets. No weights changed.

Python run 262; forwards 20. Run261 inspection failed on a module.shape attribute; no edits changed. Partial latent helper includes layer normalization, so raw/helper mismatch was not evidence of stale tokens. Comparing full posterior and changed-dose token reconstructions.

Python run 263; forwards 20. Full changed-dose token reconstruction is exact, ruling out the suspected token bug. Recovering architecture and joint raw-state feature interfaces for a total-response constrained fit. Checkpoint222 unchanged.

Python run 264; forwards 20. Architecture recovered. Gathering bounded joint-feature and calibration interfaces; planned fit constrains total attention-plus-FFN response instead of FFN contrasts alone. No edits changed.

Python run 265; forwards 20. Checking factual/cache alignment and physical-dose interfaces before joint regression. Record-derived signs remain insulin negative, NE positive; checkpoint222 remains installed and factually validated.

Python run 266; forwards 20. F81 is not aligned with current input tokens and is excluded as a regression reference. Verifying cached original/current prelude outputs and physical doses. No weights changed.

Python run 267; forwards 20. Fitted total-raw-response linear corrections from exact factual and x0/.5/2/3 tokens, using record-derived full-sensor directions and whole-stay holdout. Factual target is zero change from validated checkpoint222. Candidates retained, not installed. Scores: [(0.1, 0.039182707549094066, 0.3225326976550836), (1.0, 0.06594051023847738, 0.29038699801060247), (10.0, 0.09727936458921893, 0.28322170642726346), (100.0, 0.17708964867728097, 0.28934092855428184)]

Python run 268; forwards 20. Installed reversible total-response candidate gamma=.1, hash a8031386316e37a7; checkpoint267 retains validated checkpoint222. Held-out normalized-state RMS .03918 and contrast relative error .3225. Next factual test; accept only original value NLL +<=.005 nats.

Python run 269; forwards 21. Total-response gamma=.1 candidate passes initial factual test: original 1.63982, edited 1.64078, delta +.00096 nats on 100 tuning stays. Recovering exact rollout stay IDs and record-derived references before paired dose-rollout validation. No weights changed.

Python run 270; forwards 22. Candidate gamma=.1 factual delta +.00096 but actual rollouts (-19.551,+1.037,+1.719,-1.206) remain asymmetrically wrong for insulin. Local total-state constraints did not solve rollout behavior. Inspecting rollout levels, exact targets and retained implementation; no new edits.

Python run 271; forwards 22. Recovered exact reference shifts (-12.9555,+6.4778,+4.2797,-2.1398). Inspecting reached-row saturation, baseline drift, temporal-direction history and exact carrier helpers before choosing a reversible experiment. Current candidate unchanged.

Python run 272; forwards 22. No simple glucose ceiling: half-dose positives only 41.6%; repaired baseline drifts glucose +~10 and MAP -~3 units. Saved carrier tests are approximately symmetric. Recovering recurrence/readout replay to address downstream feedback rather than repeating scalar payload scaling. No edits changed.

Python run 273; forwards 22. Installed reversible whole-output-path scaling experiment: attention out and FFN output weight/bias deltas multiplied by .5; latent encoder and attention q/kv unchanged. This differs from prior isolated carrier payload scaling. checkpoint273 retains the factually validated gamma=.1 candidate. Next factual safeguard, then rollout if safe.

Python run 274; forwards 24. Restart recovery: whole-output-path halving scarcely changed rollout effects (-19.559,+1.026,+1.639,-1.328), refuting simple output-scale calibration. Recovering architecture, early identification evidence, helpers and factual pass 23. No weights changed.

Python run 275; forwards 24. Architecture reread: edited prelude precedes normalization and recurrent feedback. Bounded recovery of implementation/checkpoints after run274 output truncation. No weights changed.

Python run 276; forwards 24. Whole-output halving passed factual (+.001278) but provided no causal improvement; restored checkpoint273 (validated total-response gamma=.1). Recovering carrier and exact recurrence/readout implementation. No new candidate installed.

Python run 277; forwards 24. Existing posterior/FFN edits have very large norms, raising an off-record stability hypothesis (not established). Recovering original-token dose contrasts and replay interfaces to test a simpler correction around original weights. Current validated checkpoint273 unchanged.

Python run 278; forwards 24. Rebuilt exact original-posterior factual and paired changed-dose tokens, retaining original prelude features. Calibrated raw-state directions against record-only four-sensor effects using central finite differences; no gradients or weight changes. Preparing simpler original-prelude FFN correction.

Python run 279; forwards 24. Fitted closed-form, rank-two original-prelude FFN corrections on paired x0/.5/2/3 dose contrasts, with factual-zero penalty and whole-stay holdout. Original encoder, posterior, attention and recurrence remain the candidate reference. Candidate scores: [(0.01, 0.024230569015838494, {0: 0.9575096949785293, 1: 0.931515726566097}), (0.1, 0.09088678921346283, {0: 0.8420630249303332, 1: 0.8739990050461318}), (1.0, 0.19375442601086346, {0: 0.7869843424395334, 1: 0.8539955613775102}), (10.0, 0.303926772129041, {0: 0.8024107161081019, 1: 0.8386076914667373}), (100.0, 0.5285122654935647, {0: 0.7880261513641545, 1: 0.8336137886092878})]. No weights changed.

Python run 280; forwards 24. Rejected original-prelude-only regression without installation: held-out dose contrast errors 0.79-0.96 and excessive weight norms. Replayed current repaired factual and all x0/.5/2/3 dose cases for one step to distinguish local response error from rollout feedback. Results: {(0, 0.0): (array([-0.24033356,  0.42668152, -0.00966644,  0.46125793], dtype=float32), array([-0.        ,  1.99742813, -0.        ,  1.96731562])), (0, 0.5): (array([-0.24108124,  0.14408875, -0.03879547,  0.16142273], dtype=float32), array([-0.        ,  0.99871406, -0.        ,  0.98365781])), (0, 2.0): (array([ 0.5097122 , -0.27095032,  0.09243774, -0.4105072 ], dtype=float32), array([ 0.        , -1.99742813,  0.        , -1.96731562])), (0, 3.0): (array([ 0.90943146, -0.65922546,  0.10076904, -0.95158386], dtype=float32), array([ 0.        , -3.99485625,  0.        , -3.93463125])), (1, 0.0): (array([-0.4339447 ,  0.13146973, -0.31063843, -0.2071228 ], dtype=float32), array([-0.89625001, -0.        , -0.95361001, -0.        ])), (1, 0.5): (array([-0.30703735,  0.04785156, -0.13989258, -0.10696411], dtype=float32), array([-0.448125, -0.      , -0.476805, -0.      ])), (1, 2.0): (array([ 0.63018036, -0.38453674,  0.2515564 , -0.20501709], dtype=float32), array([0.89625001, 0.        , 0.95361001, 0.        ])), (1, 3.0): (array([ 1.170723  , -0.81484985,  0.45667267, -0.377594  ], dtype=float32), array([1.79250001, 0.        , 1.90722001, 0.        ]))}. Validated checkpoint273 remains installed.

Python run 281; forwards 24. Latest one-step response is weak and cross-contaminated despite oversized insulin rollout effect. Recovering exact transition and record-only identification evidence before testing accumulated feedback. No weights changed.

Python run 282; forwards 24. Architecture confirms generated-value feedback through recurrence and last-value registers. Run281 output truncated; recovering bounded helper and identification inventory. No weights changed.

Python run 283; forwards 24. Bounded recovery of exact replay interface, physiological coefficients and changed-dose caches. Prelude edits are substantial; off-record instability remains a hypothesis, not an established cause. No weights changed.

Python run 284; forwards 24. Installed reversible output-residual scale 0.02: attention out and FFN output deltas scaled relative to validated checkpoint273; posterior, z_in, q/kv unchanged. checkpoint284 preserves prior validated edit. Test whether strong contraction changes normalized off-record behavior; factual safeguard required. Hash ac68d64743844898.

Forwards 25-26: output scale .02 factual delta +.00175; dose shifts (-15.750,+1.310,+.744,-1.195), versus record references (-12.9555,+6.4778,+4.2797,-2.1398). Scale improves main readout but does not solve asymmetry.
Python run 285; forwards 26. Preserved contracted candidate in checkpoint285. Installed retained total-response FFN candidate gamma=10.0, with all other weights restored from checkpoint273. Stronger total-response constraint tested next with factual safeguard. Hash 70258d9b5c6102c7.

Python run 286; forwards 26. Restart recovery: reread architecture, recovered exact tuning IDs and recent candidate/checkpoint inventory. Gamma=10 total-response candidate remains installed; no weights changed. Factual safeguard is next.

Python run 287; forwards 28. Gamma=10 passes factual (+.00049 nats), but rollout shifts (-19.716,+1.096,+2.232,-.820) remain asymmetric. Saved as checkpoint287. Recovering carrier implementation; no new weights installed.

Python run 288; forwards 28. Recovering exact dose-carrier and total-response regression helpers to examine off-record robustness. No weights changed; checkpoint287 remains the gamma=10 candidate.

Python run 289; forwards 28. Installed reversible output-residual contraction alpha=.005 relative to checkpoint273; posterior, z_in and q/kv remain unchanged. Previous alpha=.02 passed factual (+.00175) and improved insulin x2 to -15.750 but left other readouts weak. Testing stronger contraction with factual safeguard, then paired rollout. checkpoint287 preserves gamma=10; checkpoint285 preserves alpha=.02.

Python run 290; forwards 30. Alpha=.005 rollout (-8.092,+1.383,+.104,-1.119) confirms contraction alone cannot repair all dose directions. Recovering earlier validated candidates and exact editing helpers before selecting a drug-specific reversible test. No weights changed.

Python run 291; forwards 30. Recovery output from run290 was truncated before helper inventory. Reading bounded retained-function/checkpoint inventory and earlier numerical validations. Record-derived signs and persistent effects are supported; no weights changed.

Python run 292; forwards 30. Preserved alpha=.005 in checkpoint292 and installed alpha=.01 relative to checkpoint273 for attention-out and FFN-output deltas; other checkpoint273 weights unchanged. This brackets tested alpha=.005/.02 and is a fallback calibration, not a resolution of dose asymmetry. Recovering dose-channel/readout helpers alongside factual pass29. Next factual safeguard and paired rollout.

Python run 293; forwards 32. Alpha=.01 factual delta +.00185; rollouts (-12.385,+1.361,+.415,-1.193). Main insulin result is close to record-derived reference, but other directions remain weak. Investigating whether engineered posterior policy/context coupling reinterprets generated outcome changes as dose-selection information; this is a hypothesis, not established. No weights changed; checkpoint_alpha01 retains current validated fallback.

Python run 294; forwards 32. Preserved validated alpha=.01 in checkpoint294. Estimated a local two-drug context-response subspace by central finite differences through the exact frozen recurrence (no gradients). Removed that subspace from engineered context-policy rows31(insulin)/93(NE), centering each on its active-drug training contexts. Gates, physical-dose decoder and all other weights unchanged. This tests treatment-induced policy feedback as a cause of rollout asymmetry; candidate checkpoint_policy294 requires factual and rollout validation. Singular values: [0.673505929901084, 0.28008316753851964].

Python run 295; forwards 32. Restart recovery of architecture, tuning IDs, checkpoints and record-derived evidence. No weights changed. Centered policy-context decoupling from run294 remains installed and requires factual and paired-rollout validation.

Forwards 33-34: centered context projection passed factual on 100 tuning stays (1.64168 versus 1.63982; delta +.00186). Dose shifts (-12.318,+1.412,+.418,-1.203): negligible change from alpha=.01, arguing against the tested local feedback subspace as the main cause of asymmetry.
Python run 296; forwards 34. Preserved current validated weights in checkpoint296; bounded recovery of carrier/policy implementation and record-only evidence. No weights changed.

Python run 297; forwards 34. Installed reversible flat-policy candidate from checkpoint294 (validated alpha=.01). Zeroed context columns of posterior policy rows31/93 and adjusted biases to preserve their active-drug training-context mean activations. Dose decoders, gates, and all other weights unchanged. checkpoint_flat297 preserves candidate; checkpoint296 and checkpoint294 preserve validated fallbacks. Factual and rollout tests required; no improvement assumed.

Forwards 35-36: flat-policy candidate factual delta +.00187; rollout shifts (-12.342,+1.391,+.408,-1.229). Removing all context dependence did not resolve asymmetry; rejected.
Python run 298; forwards 36. Restored checkpoint294 and installed a reversible drug-specific output graft. For attention-out and FFN-output weight/bias, added the NE-direction component of (checkpoint287 minus checkpoint294), using the dual of the two finite-difference-calibrated raw-state drug directions. Other weights unchanged. checkpoint_ne298 preserves candidate and ne_graft_deltas298 preserves increments. Requires factual and paired-rollout validation; no improvement assumed.

Python run 299; forwards 36. Recovered architecture, record schema, all tuning cohort IDs, checkpoint structures and early identification notes. No weights changed. NE graft from run298 remains unvalidated; algebraic directional isolation does not establish rollout isolation. Next: factual safeguard and paired dose rollouts, then select a validated checkpoint in the final available Python run.

Python run 300; forwards 37. NE graft rejected: factual NLL 1.65223 versus original 1.63982, delta +0.01241, exceeding +0.005; deterioration concentrated in glucose heads. Restored checkpoint294 exactly and verified all checkpoint tensors. No further Python runs remain. Selected validated alpha=.01 candidate: prior factual delta +.00185; first-cohort shifts (-12.385,+1.361,+.415,-1.193), against record-derived references (-12.9555,+6.4778,+4.2797,-2.1398). Main insulin response is close, but the remaining responses are underestimated; not a complete causal repair. Next: broader tuning factual and dose validation without further weight changes.
