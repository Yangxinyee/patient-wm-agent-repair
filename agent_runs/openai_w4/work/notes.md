# World-model weight repair

## Constraints and objectives
- Infer effects only from the supplied simulated-ICU records.
- No model backpropagation or weight optimiser.
- Primary target: insulin x2 from hour 12, glucose at eligible hours 13–24.
- Also check insulin x0.5 and norepinephrine x2/x0.5.
- Factual value NLL increase must be <= 0.005 nats.
- Budget: 300 Python runs and 50 forward requests.

## Run 1
Read architecture and schema before choosing identification or editing methods.
No assumptions about the sign or existence of drug effects.

## Run 2
Architecture: action doses enter encode_action, posterior z, and the recurrent
prelude. Measurement heads additionally use a last-value register.
Loaded training and tuning records. Training stays: 30000;
tuning stays: 400. No effect conclusions or weight edits yet.

## Run 3; forward requests used: 1
Original dose-edit predictions on tuning stays 2020000000–2020000099:
insulin x2 -0.382; insulin half +0.209 mg/dL;
NE x2 +0.103; NE half -0.307 mmHg.
These are predictions, not estimated truth.
Built hourly panels, preserving measurement missingness and filling absent doses
with zero. Readings are at integer hours; infusions start at hour +0.0833.
Next step: identify dose-policy variation and test adjusted outcome effects.

## Run 4; forward requests used: 2
Original factual NLL on first 100 tuning stays: 1.59524 (4,903 readings).
Insulin threshold policy best explained by finger_first;
classification error 0.003178.
Computed local-linear RD estimates at 150.5, 180.5, 220.5, 260.5,
with stay-clustered uncertainty and pre-outcome/observation checks.
RD table:
 cutoff  bw      n  dose_jump  outcome_jump  effect_per_rate  cluster_se
  150.5   5  31178     0.1914       -1.0127          -5.2916      1.6000
  150.5  10  61110     0.4492       -1.2418          -2.7645      0.5122
  150.5  15  90105     0.5762       -1.3898          -2.4122      0.3402
  150.5  20 118034     0.6575       -1.5475          -2.3536      0.2653
  180.5   5  38087     0.5698       -0.7370          -1.2935      0.5515
  180.5  10  75667     0.7794       -1.1972          -1.5359      0.2877
  180.5  15 112580     0.8600       -1.4093          -1.6386      0.2149
  180.5  20 148824     0.9038       -1.5463          -1.7108      0.1791
  220.5   5  13465     0.6094       -1.9360          -3.1771      1.0528
  220.5  10  27007     0.8056       -1.8092          -2.2459      0.5494
  220.5  15  41128     0.8857       -1.9515          -2.2034      0.4082
  220.5  20  55451     0.9242       -2.0053          -2.1697      0.3407
  260.5   5   2121     0.4853       -1.7844          -3.6771      3.6196
  260.5  10   4328     0.7606       -2.5573          -3.3620      1.6647
  260.5  15   6845     0.8875       -2.7845          -3.1374      1.1748
  260.5  20   9743     0.9343       -2.9469          -3.1542      0.9651

## Run 5
RD at the initial insulin threshold was complicated by rounded readings and
a discontinuous next-reading probability (about 30% to 100%).
Refined RD excludes the rounded threshold itself.
Refined RD:
 cutoff  bw      n  dose_jump  outcome_jump  effect_per_rate  cluster_se
  150.0   5  23532     1.0369       -2.4967          -2.4078      0.4769
  150.0  10  53539     1.0411       -2.1079          -2.0248      0.2900
  150.0  15  82589     1.0439       -2.0414          -1.9555      0.2283
  150.0  20 110599     1.0452       -2.1108          -2.0195      0.1941
  180.0   5  30810     1.0416       -1.8195          -1.7468      0.3929
  180.0  10  68730     1.0476       -1.7854          -1.7044      0.2402
  180.0  15 105934     1.0452       -1.8119          -1.7336      0.1902
  180.0  20 142517     1.0440       -1.8588          -1.7805      0.1641
  220.0   5  10878     1.0678       -3.1426          -2.9430      0.7643
  220.0  10  24707     1.0534       -2.2490          -2.1350      0.4629
  220.0  15  39001     1.0530       -2.2109          -2.0996      0.3641
  220.0  20  53461     1.0493       -2.1822          -2.0797      0.3138
  260.0   5   1722     0.9170       -4.7440          -5.1732      2.5458
  260.0  10   3976     1.0306       -3.9808          -3.8625      1.3933
  260.0  15   6526     1.0794       -3.8052          -3.5254      1.0430
  260.0  20   9462     1.0751       -3.7279          -3.4675      0.8812
Two-fold, stay-separated history-adjusted dose/outcome residual regressions:
          drug         target      n     beta  cluster_se  dose_R2  dose_residual_SD  outcome_residual_SD
       insulin glucose_finger 391466 -2.04924     0.04571  0.80748           0.48910             12.20504
norepinephrine   map_arterial 330203  3.17146     0.08848  0.92657           0.09261              4.31153
These estimates require dose innovations to be unconfounded conditional on
recorded history; threshold agreement and placebo tests are used as checks.
No weights edited.

## Run 6
Previous forward request failed because of duplicate stay IDs; it was not counted.
Re-read architecture/schema and inspected retained kernel objects and tensor shapes.
Constructed a verified list of 100 unique tuning stay IDs.
Drug-effect estimates remain provisional; no weights edited.

## Run 7
Architecture inspection: drugs enter posterior and recurrent prelude; value heads also have a same-code last-reading register.
Considering targeted readout repair, conditional on states retaining dose information.
Run 6 output was truncated; obtained compact remaining tensor/vocab/object summaries.
No weights edited.

## Run 8; forward passes used: 3
Original states obtained on first 100 tuning stays, all bundles, 4531 rows in /work/fwd_4/states_original.npz.
Aligned states to hourly records; tested held-out-stay ridge rate decoding from out, out+h, and out+h+z.
Inspected normalisation, last-numeric dimensions and retained DML objects.
No weights edited.

## Run 9
Run 8 rate decoding with ridge alpha=10: insulin R2 0.513 (out), 0.657 (full); norepinephrine R2 0.273 (out), 0.485 (full).
Tested weaker ridge penalties and conditional dose-innovation decoding with stay-separated validation. This distinguishes policy information from usable administered-dose information.
          drug features    alpha  rate_R2  innovation_R2  innovation_RMSE  innovation_SD
       insulin      out  0.00001  0.58005       -0.02652          0.62983        0.62164
       insulin      out  0.00100  0.64883        0.14715          0.57408        0.62164
       insulin      out  0.10000  0.57190       -0.00724          0.62389        0.62164
       insulin      out 10.00000  0.51261       -0.01938          0.62763        0.62164
       insulin        z  0.00001  0.49403        0.09081          0.59274        0.62164
       insulin        z  0.00100  0.52381        0.06004          0.60269        0.62164
       insulin        z  0.10000  0.39526        0.04280          0.60819        0.62164
       insulin        z 10.00000  0.36742        0.04004          0.60907        0.62164
       insulin     full  0.00001 -0.61610       -3.95211          1.38336        0.62164
       insulin     full  0.00100  0.41363       -0.64819          0.79807        0.62164
       insulin     full  0.10000  0.59641       -0.00751          0.62397        0.62164
       insulin     full 10.00000  0.65694        0.16565          0.56782        0.62164
norepinephrine      out  0.00001  0.01072       -0.67818          0.17887        0.13808
norepinephrine      out  0.00100  0.46945        0.02358          0.13644        0.13808
norepinephrine      out  0.10000  0.47133        0.11145          0.13016        0.13808
norepinephrine      out 10.00000  0.27287        0.00654          0.13763        0.13808
norepinephrine        z  0.00001  0.38022       -0.29751          0.15728        0.13808
norepinephrine        z  0.00100  0.37142        0.06193          0.13373        0.13808
norepinephrine        z  0.10000  0.17373        0.03007          0.13599        0.13808
norepinephrine        z 10.00000  0.09161        0.02544          0.13631        0.13808
norepinephrine     full  0.00001 -2.24835       -9.78801          0.45352        0.13808
norepinephrine     full  0.00100 -0.59974       -3.23869          0.28428        0.13808
norepinephrine     full  0.10000 -0.05638       -0.73221          0.18173        0.13808
norepinephrine     full 10.00000  0.48529        0.00615          0.13765        0.13808
Inspected norepinephrine assignment versus latest recorded MAP and confirmed DML outcome alignment. No edits.

## Run 10
Dose innovations decode weakly from original states (best approximately 0.17 insulin R2 and 0.11 norepinephrine R2). Readout-only repair is not yet justified; investigating action-pathway repair.
Implemented candidate CPU head reconstructions using exported states; tested activation, measurement type index and last-numeric ordering against original forward predictions.
activation  type              last_pattern  mean_RMSE  nll_RMSE  mean_nll
      silu     1           has_age_numeric    7.97621   0.57902   1.74300
      gelu     1           has_age_numeric    8.56412   0.63368   1.75722
      relu     1           has_age_numeric   10.64200   0.77992   1.83793
      relu     1           numeric_has_age   10.01018   1.13823   2.03353
      gelu     1           numeric_has_age   10.66927   1.18053   2.10923
      silu     1            four_then_zero    9.01266   1.21512   1.97964
      silu     1           numeric_has_age   11.37136   1.34605   2.25092
      silu     1 has_age_zrate_zdose_flags    8.88821   1.35029   2.08227
No edits.

## Run 11
Run 10 head reconstruction failed (best mean RMSE 7.98, NLL RMSE 0.579); not trusted for editing.
Checked latent source, gap scale, register flags/status and mixed activations.
core_act  gap_factor latent  value_flag_col status head_act  mean_RMSE  nll_RMSE  mean_nll
    silu         1.0   mean               3   zero     gelu    7.65668   0.55623   1.72939
    silu         1.0   mean               3   zero     silu    7.97621   0.57902   1.74300
    silu         1.0   zero               3   zero     gelu    6.82061   0.59083   1.72546
    gelu         1.0   mean               3   zero     gelu    8.56412   0.63368   1.75722
    silu         1.0   zero               3   zero     silu    7.60936   0.63491   1.75300
    gelu         1.0   mean               3   zero     silu    9.17698   0.68173   1.78442
32-draw prior-average metrics: {'mean_RMSE': 7.68521438909105, 'nll_RMSE': 0.5575281372041756, 'mean_nll': 1.7329300033856052}
No edits.

## Runs 12–13
Register-layout search improved reconstruction to mean RMSE 2.769747 and NLL RMSE 0.290933, with numeric column 6, age column 3, flag column 0, and SiLU. Run 12 then failed with pandas Flags KeyError; use explicit column indexing rather than attribute access. Reconstruction remains insufficiently accurate for editing. Re-read architecture/schema and inspect retained objects before further tests. No weights edited.

## Run 14
Architecture explicitly specifies the four-field last_numeric register. Run 13 output was truncated; collected compact vocabulary, tensor and forward-file metadata to verify the CPU head and state alignment. No weights edited.

## Run 15
Investigated eight-dimensional last_numeric packing (architecture's description is abbreviated), reproduced the best register candidate using explicit column indexing, and examined alignment and residuals by event. No weights edited.

## Run 16
Verified that register ages are already log(seconds+1)/10. Tested status corrections with the best register layout and compared remaining head mismatch with four-draw prior variability when the latent-head prefix could be identified. No weights edited.

## Run 17
Status corrections did not improve reconstruction. Four-draw prior variability (mean RMS 1.06; NLL RMS 0.123) was too small to explain the mismatch. ReLU outperformed SiLU/GELU with the current register. Jointly searched gap/prior/head activations, prediction-gap scaling and code-query type. Best provisional configuration: ('gelu', 'gelu', 1.0, 'gelu'). No weights edited.

## Run 18
Joint search favoured GELU for gap/prior/value head, gap factor 1, and measurement query type 1. Re-searched register packing with this view, reconstructed all 4,903 readings and assessed latent noise. Provisional head configuration: {'gap_activation': 'gelu', 'prior_activation': 'gelu', 'head_gap_factor': 1.0, 'head_activation': 'gelu', 'type': 1, 'zcol': 6, 'agecol': 3, 'flags': (0,)}. No weights edited.

## Run 19
Corrected head/register reconstruction has residual errors comparable to four-draw prior variability. Began longitudinal causal diagnostics: dose innovations instrument later outcomes, explicitly accounting for covariance with subsequent recorded doses. AR persistence fits are provisional because later observation selection and nuisance misspecification can bias these moments: {'insulin': {'beta': -2.0492419728332494, 'rho': 0.9259212188520857, 'weighted_loss': 189.9550943168164, 'n_moments': 6}, 'norepinephrine': {'beta': 3.171455470405459, 'rho': 0.6834209822002549, 'weighted_loss': 22.851297233717364, 'n_moments': 6}}. No weights edited.

## Run 20 (restart)
Recovered architecture, record schema, earlier notebook context, retained objects, and original rollout summaries. The provisional one-exponential impulse model is not accepted as truth; dose-innovation identification and flexible lag responses need checking before a weight repair. No weights edited.

## Run 21
Run 20 output was truncated. Retrieved compact retained-object inventory, action/value tensor shapes and original rollout effects. Explicit action pathway and same-code measurement shortcut suggest a targeted correction rather than global output recalibration. No weights edited.

## Run 22
Recovered remaining retained objects. Deconvolved run-19 closed-loop dose-innovation moments into a provisional six-lag direct-response kernel and translated it to the tuning forward's eligible hours. This is a diagnostic, not accepted ground truth: horizon selection, nuisance misspecification and response heterogeneity remain possible. No weights edited.

## Runs 23–24
Run 23 failed before diagnostics due to NumPy/Tensor mixing. Run 24 used forward-only PyTorch to reconstruct value-head features and test stay-separated dose-innovation decoding. Head check: {'mean_RMSE': 1.5699389659718386, 'nll_RMSE': 0.1964863291754357, 'mean_NLL': 1.5826622247695923}. No weights edited.

## Run 25; forward passes used: 4
Built reusable head dataset and feature functions, verified tuning alignment, fit insulin-innovation decoders on independent training states with MAP scores penalised toward zero, and checked tuning transfer plus approximate factual costs. No weights edited. Selected decoder alpha: 10000.

## Run 26 (restart)
Run-25 decoder failed independent transfer (negative innovation R2, approximately zero covariance). Low approximate factual cost does not justify that repair. Recovering architecture and retained analyses; no weights edited.

## Run 29
Recovered baseline shifts: insulin x2 -0.382, half +0.209; NE x2 +0.103, half -0.307. Immediate history-adjusted estimates are insulin -2.049 per rate and NE +3.171 per rate. Six-lag deconvolution suggests an insulin hump followed by decay rather than the earlier single-exponential fit; long-horizon truth remains provisional. No edits.

## Runs 30–31
Run 30 stopped on absent rate_unit/dose_unit columns; no edits. Reconstructed isolated action embeddings, action encoder and z_post using GELU, testing unit/rate-unit/KQ/bin conventions against 133 exported posterior means. Best errors: {'insulin': 0.13187898695468903, 'norepinephrine': 0.13511121273040771}. No weights edited.

## Run 32
Drug-posterior reconstruction remains inaccurate (RMSE approximately 0.13 versus posterior RMS 0.248). Recovering the exact architecture and retained data before choosing a repair; no weights edited.

Run 37: recovered longitudinal diagnostics and action-reconstruction arrays. Testing whether the empty measurement encoder, rather than zero e, explains posterior mismatch. No edits.

Run 38: tested zero versus learned empty measurement encodings against isolated action posterior states; checked the pre-action context from exported states. No edits.

Run 39: empty measurement encoder resolved the major posterior mismatch. Searched exact drug categorical and bin conventions; retained best atom recipes. No weights edited.

## Run 40 (restart)
Recovered architecture, schema, retained reconstruction and causal objects. Run-39 isolated drug posterior reconstruction was accurate to approximately 7.4e-5 RMSE after including the learned empty-measurement encoding. This validates that component, not the causal effects or a weight edit. No edits.

## Run 41
Architecture separates the explicit action encoder, recurrent transition, and same-code last-reading shortcut. Recovering recent reconstruction objects and initial causal diagnostics with compact output. No edits.

## Run 42
Recovered causal-diagnostic objects, fixture schema and remaining transition metadata. Initial insulin threshold estimates around -2 per rate do not by themselves identify the 12-hour intervention effect. No edits.

## Run 43
Immediate estimates agree approximately with insulin -2 and NE +3.17 per rate. Recovering exact action fixtures and longitudinal objects with bounded output; long-horizon causal targets remain provisional. No edits.

## Run 44
Computed controlled half/double-dose perturbations through the reconstructed singleton action encoder and posterior, holding pre-action context fixed. Retrieved retained longitudinal moment tables and nuisance-model structure. No edits; component sensitivity does not establish rollout causality.

## Run 45
Controlled dose changes move action encodings and posterior means substantially, so the input/posterior is not simply dose-insensitive. Implemented general atom embedding and tested singleton measurement encodings against exported posterior means as a prerequisite for reconstructing memory and recurrent transitions. No edits.

## Run 46 (restart)
Drug atoms reconstruct accurately; singleton measurement posterior errors remain 0.07–0.12 RMS. Recovering architecture and retained diagnostics before selecting a repair. These component checks do not establish causal effects. No edits.

## Runs 47–48
Recovered threshold and innovation evidence for immediate effects, but sustained response is not yet accepted as ground truth. Inspecting lagged moments and observation selection before choosing a target. No edits.

## Run 53
Run 52 failed by requesting a value embedding for hospital_admission. Recovering architecture and retained objects before correcting event-type handling. No causal conclusion follows from that exception; no weights edited.

## Run 54
Architecture recovered; run-53 object output truncated. Retrieving compact causal/reconstruction inventory rather than repeating the full inventory. No weights edited.

## Run 55
Retrieved retained threshold, innovation and longitudinal-moment diagnostics to select an evidence-based intervention target. No weights edited.

## Run 56
Immediate innovation estimates: insulin -2.049 +/-0.046, NE +3.171 +/-0.088 per rate; insulin RD broadly agrees. Six-lag kernels show non-single-exponential persistence. Inspecting encoding recipes and residuals for independent causal and repair validation. No weights edited.

## Run 57
Corrected measurement units/bin sides from run-52 best recipes (posterior RMSE about 2.5e-5); learned size-bias empty-action encoding is essential. Checking exact encoder and causal-panel alignment. No weights edited.

## Run 59
Run-58 count check found 1,392 mismatched bundles; posterior test stopped because pre_context returns a tuple. Testing whether infusion-end atoms explain all count differences. No weights edited.

## Run 60
Added inferred infusion END atoms (kind 2) to world encoder, not action encoder; checked exact bundle counts and complete posterior reconstruction. Head innovation probes have modest independent insulin R2 (~0.14), so a factual-preserving correction may be feasible but is not validated. No weights edited.

## Run 61 (restart)
Run-60 posterior reconstruction is exact for most multi-world and END bundles, but 100 count mismatches and concentrated NE/single-world errors remain. Recovering transition code and exception structure before committing a targeted edit. No weights edited.

## Run 65
The 100 count discrepancies are two visit END markers per stay at 25.1667h. Tested float32 bin-edge/input agreement and added visit ENDs. Posterior RMSE=5.636162e-05; counts now have 0 mismatches. No weights edited.

## Run 68 (restart)
Run-67 reconstructed recurrent transition matches exported h (RMSE 1e-5) and out (RMSE 1.4e-7) with all current events, unnormalised memory, and both age updates. This validates forward reconstruction, not causal identification. Recovering retained functions/data to design and validate a targeted repair. No weights edited.

## Runs 69–74
Recovered immediate estimates and six-lag kernels. Implemented reusable exact transition: out RMSE 1.4e-7, h RMSE 1.0e-5, posterior RMSE 5.9e-5. Constructed isolated half/double drug state changes and matched next-reading head features on 100 training stays. No weights edited; controlled feature changes are a repair design diagnostic, not causal evidence.

## Run 75 (restart)
Run-74 isolated next-reading sensitivities are weak for insulin and nonmonotonic for NE despite exact state reconstruction. Recovering head/perturbation objects before fitting a factual-regularized repair. No weights edited; rollout targets still require record-based validation.

## Runs 76–77
Source inspection failed because notebook functions lack retrievable source. Recovering metadata and retained arrays instead. No edits.

## Run 78
Recovered paired head perturbation arrays. Checking longitudinal record-derived targets before using immediate effects in a head repair. No edits.

## Run 79
Record-based immediate effects remain insulin -2.049/rate and NE +3.171/rate; deconvolved kernels show additional persistence. Preparing a closed-form head-feature regression with independent factual validation, not treating one-step fit as rollout success. No edits.

## Run 80
Fit closed-form scalar calibration regressions using factual bin residuals and paired dose-feature response constraints. Compared independent tuning NLL and nonlinear one-step responses over a ridge/constraint grid. No weights saved.

## Run 81
Head-only rank-one calibration cannot restore direct effects within the factual budget: meaningful increases cost far above 0.005 nats. No candidates saved. Investigating posterior action innovations and a targeted transition repair instead of globally tilting the value head.

## Run 82
Tested posterior-hidden dose-innovation regressions with 60/40 stay-separated validation and controlled half/double response constraints. No weight edits; decoder transfer and non-action leakage determine whether a posterior repair is plausible.

Runs 83–85: recovered architecture and initial record evidence. Original tuning dose shifts are small; immediate record-based effects are insulin about -2 and NE about +3.17 per rate. Neither identifies the sustained 12-hour effect alone. No edits.

Run 87: tested raw posterior-input dose-innovation decoders and action-presence gates with stay-separated validation. No weights saved. Six-lag insulin kernel has a delayed peak, unlike a single exponential.

Runs 88–89: insulin action gate has perfect held-out classification under all four controlled dose inputs. Two-pole kernel fits six insulin lags closely but longer-tail targets remain provisional. Corrected a pandas shift-column collision. No edits.

## Run 90
Two-pole kernel targets remain provisional, not established truth. Recovering architecture and retained causal/repair fixtures before validating sustained effects and choosing a targeted edit. No edits.

## Run 91
Architecture confirms an explicit action/posterior pathway. Inspecting retained innovation and action-gate validation to assess a localized posterior repair; sustained causal targets remain provisional. No edits.

## Run 92
Inspecting held-out raw-input innovation regressions and gate margins, plus exact posterior tensor dimensions. No weights edited.

## Run 93
Raw insulin decoder achieves held-out R2 about 0.62 with half/double response slopes about 0.92/0.89. NE decoder/gate are materially weaker. Prioritizing a selective insulin latent repair, with explicit leakage and factual validation; no weights edited.

## Run 94
Constructed (not saved) two gated insulin-innovation posterior units. Original removed-unit outputs are compensated by a closed-form regression on retained hidden units. Testing held-out latent fidelity and non-action leakage before selecting an outcome-response direction. No edits.

## Run 95
Two posterior hidden units can be repurposed with compensated mean-latent error about 2e-7 RMS. Checking action-flag leakage and finite-difference latent-to-reading sensitivities to derive an insulin-specific response direction by closed-form regression. No weights edited.

## Run 96
Small posterior-latent perturbations have essentially zero reading sensitivity. Testing larger perturbations and near-null latent directions to distinguish an ignored pathway from weak z_in transfer. No weights edited.

## Run 98
Run 97 confirmed its proposed candidate was not executed and no edits exist. Recovering architecture and recent retained diagnostics to locate the nearly inactive latent pathway before selecting a repair; no weights changed.

## Run 99
Architecture recovered. Inspecting latent-transfer magnitude and retained sensitivity tests; the action also has a deterministic transition path, so posterior insensitivity alone does not identify the repair location. No edits.

## Run 100
z_in has ordinary-sized weights despite negligible reading finite differences. Checking the perturbation wrapper and reading-state plumbing before inferring that the learned transition ignores z. No edits.

## Run 101
Verified posterior injection reaches z but changes out only about 1e-6 RMS for a +5 latent perturbation. The latent-to-state route is nearly inactive. Inspecting prelude/state scales and the retained gated insulin innovation before constructing a selective stronger connection. No edits.

## Run 102
Captured prelude slot representations for the retained posterior-training bundles and tested stay-held-out linear decoding of the gated insulin innovation. This tests a simpler rank-one recurrent-matrix repair before modifying attention. No weights edited.

## Run 103
Prelude innovation decoding failed (held-out insulin R2 only ~0.05–0.10), rejecting the simple B_in shortcut. Captured exact attention inputs and tested whether a redundant head can be compensated before creating an explicit gated-innovation channel. No edits.

## Run 104
Constructed an unsaved explicit innovation-reading attention head (head 1), with compensated original attention output and gated posterior innovation in a dedicated latent direction. Tested token isolation, scalar transfer, and held-out baseline compensation. No edits saved.

## Runs 105–108
Recovered candidate architecture and interfaces after restart. Head 1 is the most compensable attention head (held-out output RMS about 0.095); candidate remains unsaved pending reading-level and GPU factual tests. Immediate record estimates remain insulin -2.049/rate and NE +3.171/rate; sustained rollout targets are not inferred from these alone.

## Runs 109–115 recovery
Run-112 controlled channel decodes q roughly monotonically, but has factual/non-action leakage 0.18–0.24 RMS. No weights saved. Runs 113–115 recover retained candidate interfaces; six-lag targets remain provisional and must not be equated with sustained truth.

## Run 117: first saved candidate
Saved amplitude-80 explicit insulin-innovation channel (9 tensors): z_post hidden/output, z_in, and prelude attention q/kv/out weights. No B_in response connection yet. This isolates factual cost of rewiring; forward factual validation is next. Original first-100 tuning NLL was 1.59524; allowed increase 0.005.

## Forward 5 / Run 118
Channel-only factual validation, first 100 tuning stays: original 1.59524, edited 1.59518 (delta -0.00006). Now testing B_in response directions against record-based insulin immediate coefficient -2.04924, with MAP target zero. Finite perturbations are design checks, not causal evidence.

## Run 119
Run-118 response directions are approximately linear and glucose-selective. Recovering exact retained wiring and earlier record evidence before connecting the response; finite differences are not sustained causal validation. No additional weights edited.

## Run 123: provisional insulin response connection
Recovered exact interfaces: rW is the CPU tensor dictionary; transition73 is the verified one-bundle transition; transition_with_b110 injects an additive core.b_in output; probe_means95 evaluates the next reading. chosen_channel116['decoder'] has shape (192,), decoding prelude E per slot. directions118[0.01] is the smaller glucose-selective response direction (norm 23.695), calibrated to immediate record estimate -2.04924/rate. Saved core.b_in.weight = original + outer(direction, decoder), scale 1.0, in addition to the nine channel edits. This remains provisional pending independent GPU factual and rollout tests; sustained kernels are not yet treated as established ground truth.

## Forward 6 / Run 124: reject rank-one connection
Full factual first-100 tuning NLL: original 1.59524, connected edit 1.92945, delta +0.33421. All four reading types worsened. Dropped core.b_in.weight; nine channel-only edits remain. Immediate finite-difference calibration did not control recurrent persistence/feedback. Next: reconstruct complete factual trajectories and design a dynamically constrained, reading-selective connection.

## Run 125
Implemented replay_factual125 using stay-separated sequential updates of repair_bank['x']; original full-state RMSE=1.01486e-05. Retained weights_channel125, weights_rejected125 and replay states/q captures for dynamic diagnosis. No additional saved edits.

## Run 126
The rejected connection has large recurrent amplification, while the channel-only replay remains near the original. Recovering exact architecture, fixtures and causal evidence before designing a dynamically bounded repair. No additional weights edited.

## Run 127
Architecture confirms that the repair enters a recurrent pathway. Recovering compact replay interfaces and initial causal evidence; the previous diagnostic output was truncated. No additional weights edited.

## Run 128
Tested scaled versions of the rejected rank-one response with complete factual replay. Results retained in scale_results128; no additional tensors saved. Recovering record evidence and reading interfaces for dynamic calibration.

## Run 129
Measured complete-replay factual NLL for response scales 0.1–0.9 on 100 training stays, with common prior draws. Results in nll_scales129; no additional tensors saved. Stability and factual likelihood must both hold before GPU validation.

## Run 130
Saved provisional core.b_in.weight response at scale 0.4. Complete training replay dNLL=-0.000229 and out RMSE=0.00561; scale 0.6 is unstable. This is not yet rollout validation or proof of sustained causal accuracy. Ten tensors now edited.

## Forward 7–8 / Run 131
Scale-0.4 edit tuning rollouts: insulin x2 -1.712, half +2.202; NE x2 +0.095, half -0.299. Asymmetry remains unexplained. Independent tuning stays 100–199 factual NLL: original 1.59931, edited 1.59848 (delta -0.00083). Recovering lag estimates and rollout detail; these predictions are not truth.


## Run 132
Recovered architecture, retained replay/repair interfaces and earlier identification evidence. Current GPU rollouts remain predictions, not truth; immediate DML coefficients alone do not establish sustained intervention effects. No weights changed.


## Run 133
Run-132 output was truncated. Compact recovery of causal evidence, function dependencies and edit magnitudes; no weights changed.


## Run 134
Recovered dynamic-identification notes and compared retained-kernel convolutions with the latest rollout on matching stay-hours. These convolutions are provisional record-based targets, not supplied truth. No weights changed.


## Run 135
Corrected the NE provisional-target label mismatch (the tool uses ne_x2/ne_half). Recovered innovation-model, impulse-moment and feedback fixtures for independent causal and dynamic validation. No weights changed.


## Run 136
Independently re-estimated six longitudinal dose-innovation moments with new stay-separated outcome nuisance regressions. Retained cluster moment arrays and direct deconvolution in lp_new136. This checks nuisance sensitivity; linear, time-invariant direct effects and innovation exogeneity remain identifying assumptions. No weights changed.


## Run 137
Extended independently fitted innovation local projections through 12 lags; computed 300 stay-cluster multiplier-bootstrap direct-kernel and matched-readout intervals. Tested a quadratic immediate dose term with cross-fitted dose-squared nuisance estimates. These estimates still depend on sequential exogeneity, sufficient history and an additive transferable impulse response. No weights changed.

## Run 138
Twelve-lag record estimates support sustained insulin and NE responses far larger than the current rollout edit. Quadratic immediate terms raise extrapolation concerns. Recovering architecture and retained interfaces to design a bounded dynamic repair; no weights changed.

## Run 139
Inspecting existing stability/NLL scale tests and exact transition/replay interfaces before selecting a bounded response repair. No weights changed.

## Run 140
Extracted numerical scale stability/NLL summaries and compact transition bytecode. No weights changed.

## Run 141
Scale 0.5 training replay is stable enough to test: mean NLL 1.6355845, original 1.6355364. Saved B_in response scale 0.5; scale 0.4 backed up in backup_b141. Awaiting independent GPU factual and rollout validation; no causal claim from replay alone.

## Run 142
GPU scale-0.5 factual first-100 tuning dNLL=+0.00037. Rollouts: insulin x2=-2.252, half=+2.748; NE x2=+0.090, half=-0.290. Scaling alone remains inadequate. Investigating input dose compression versus response dynamics before another edit. Scale 0.5 remains saved.

## Run 143
Recovering exact action construction and dose/state alignment to test whether the innovation channel compresses doubled doses. No saved weights changed.

## Run 144
Compact recovery of architecture, channel parameters and retained interfaces. Run-143 truncation did not diagnose dose compression. No weights changed.

## Run 145
Located retained action/channel functions and compact longitudinal-effect estimates. Architecture allows both encoder compression and recurrent-dynamics failure; neither is established yet. No weights changed.

## Run 146
Recovered lag-kernel/readout estimates and action encoder interfaces. Immediate dose coefficients are not equated with sustained intervention effects. No weights changed.

## Run 147
Probed actual action encodings and decoded prelude channel under dose multipliers 0, 0.5, 1, 2, 3 and 4, using fixed factual contexts. This is a model-mechanism diagnostic, not causal evidence. No weights changed.

## Run 148
Repeated channel summaries on contexts finite under all multipliers. Run-147 insulin NaN summaries were invalid; no compression conclusion was drawn. No weights changed.

## Runs 149–150
Run 149 stopped at an interface error (context return is a tuple). Verified return structures and recovered original/edited rollout comparisons. No weights changed; next diagnostic is a bounded multi-step impulse test.

## Run 151
Recovered exact transition/replay interfaces, architecture, retained channel and causal-kernel objects for a bounded impulse diagnostic. Run 150 verified return structures only; it did not establish input compression or causal truth. No saved weights changed.

## Run 152
Printed compact causal moment and transition-interface diagnostics after run 151 output truncation. No weights changed.

## Run 155
Run 154 recovered estimated equal-stay doubling targets (insulin glucose -13.2008; NE MAP +3.6087) but failed by treating Python's builtin next as an index array. Computed replay baseline before diagnostics and inspected exact indexing rather than guessing it. No weights changed.

## Run 156
Recovered architecture and retained identification/channel objects. Run-155 insulin dose-channel slopes are near proportional; NE channel slopes are weak and inconsistent. This diagnoses encoding only, not drug effects. No weights changed.

## Run 157
Compact recovery of record-based effect estimates and replay interfaces. No weights changed; recurrent persistence versus amplification still needs direct testing.

## Run 158
The 12-lag record kernels support sustained insulin lowering and NE raising effects under innovation-exogeneity/additive-kernel assumptions. Recovering exact replay/action construction for a one-dose persistence test. No weights changed.

## Run 159
Tested isolated dose increments with exact action-encoding parity checks and complete factual replay. Later readings/registers remain factual, so this is a state-persistence diagnostic rather than a free-rollout or causal-effect estimate. Results retained in impulse_results159. No weights changed.

## Run 160
Run-159 factual replay showed insulin impulse responses about [-0.989,-0.148,-0.041,-0.015] at lags 1–4 versus sustained record kernels; NE responses near zero. Tested retained connection directions for factual stability and isolated-dose response. These are mechanism diagnostics, not counterfactual truth. No weights changed.

## Run 162
Run-161 training scales through 2.5 remained within the factual tolerance, but scales >=3.5 showed recurrent amplification. Its Tensor.copy error prevented the subsequent impulse diagnostic. Recovering architecture and retained interfaces before further edits; no saved weights changed.

## Run 163
Recovered architecture in run 162, but its output truncation hid the retained interfaces. Using compact inspection now. Record threshold and dose-innovation evidence supports insulin lowering; long-horizon magnitude still requires dynamic identification. No saved weights changed.

## Run 164
Inspecting exact action/replay dependencies and retained connection tests. Defined tensor-safe copying for the interrupted impulse experiment. No saved weights changed.

## Run 165
Recovered replay's action input U65 and treatment-channel capture. Compact inspection distinguishes connection candidates and recovers the prediction head for lagged impulse testing. No saved weights changed.

## Run 166
Reconstructed the ridge-0.1 connection direction exactly and tested a +0.5 insulin impulse at submitted/1/2/2.5 strengths. Later readings remain factual: fading responses do not refute sustained free-rollout effects mediated by generated readings. Encoding parity checked; results retained in impulse_tests166. No saved weights changed.

## Run 167
Saved provisional ridge-0.1 B_in response at scale 2.0 (channel_weights160 plus 2*outer166); previous saved edits backed up in backup_before167. Training replay dNLL for this direction/scale was +0.001832. Run-166 impulse response is immediate and reading-mediated; actual free-rollout validation is next. No inference of truth from model impulse responses.


## Run 168
GPU factual pass 12 (/work/fwd_13), current scale-2 connection edit: original NLL 1.59164, edited 1.59182, delta +0.00018 on 100 tuning stays. Recovered matching rollout IDs and architecture/identification context. No weights changed. Free-rollout efficacy and broader factual stability remain untested.


## Run 169
GPU pass 13, scale-2 edit: insulin x2 -8.992, half +11.09; NE x2 -0.099, half -0.468 (100 tuning stays). Original: -0.428, +0.096, -0.096, -0.488 respectively. Insulin free-rollout sensitivity improved, but asymmetry and absent NE repair make blind shared-scale increases unjustified. Recovering retained nonlinear identification and channel construction. No weights changed.


## Run 170
Recovered immediate record estimates from independent run-136 nuisance fits: insulin -2.0468 (SE .0483), NE +3.1429 (SE .0960), with close fold agreement. These require innovation exogeneity, not medical assumptions. Inspecting nonlinear identification and separate channel construction before a targeted NE/asymmetry repair. No weights changed.


## Run 171
Run-117/123 construction is explicitly insulin-only, explaining the lack of NE repair. Run-137 quadratic immediate fits: insulin linear -1.6589, quadratic -.0740; NE linear +2.3983, quadratic +.5930. These local observational fits do not establish doubled-dose extrapolation. Inspecting occupied latent/attention dimensions and channel-building interfaces to construct an independent NE innovation channel. No weights changed.


## Run 172
Current insulin channel replaces posterior hidden rows 96 and 184 and prelude attention head 1 (48-dimensional). Investigating an independent NE signal through the already reserved head, avoiding removal of another attention head. No weights changed.


## Run 173
Checked latent-map rank, reserved attention value/output directions, original training-bank IDs and exact capture/action interfaces for a shared-head NE channel. Existing saved weights remain unchanged.


## Run 174
Recovered architecture, retained insulin-channel construction objects, and record-identification summaries before designing a separate NE channel. No weights changed. Free-rollout efficacy and factual preservation remain the validation criteria, not the presence of spare dimensions.


## Run 175
Inspected retained channel algebra and reference-bank interfaces. The current channel is insulin-only; a separate NE signal is being considered within the same reserved head. No weights changed; statistical identification assumptions are not replaced by mechanism diagnostics.


## Run 176
Inspected posterior-row replacement, innovation-fit arrays and exact capture interfaces to reproduce the channel for NE. No weights changed. A successful extension requires held-out decoding and absence of non-drug leakage before free-rollout/factual validation.


## Run 177
Recovered compact gating/MLP diagnostics and record-target interfaces after earlier output truncation. No weights changed. NE encoding will be checked against actual recorded doses and held-out bundles rather than medical priors.


## Run 178
Recovered NE raw-decoder/gate fits and matched kernel targets. Earlier NE decoding was weaker than insulin, so an independent dose-sensitivity test is required before using it in a shared-head channel. No saved weights changed.


## Run 179
Synchronized the CPU weight dictionary to submitted edits; previous CPU B_in was stale. Backed up submitted edits. Inspected NE decoder/gate candidates before selecting a held-out dose-sensitive channel. No saved weights changed.


## Run 180
Refitted NE action gates on factual and perturbed encodings, evaluating on held-out stays. Existing gate misses under half doses motivate this change. Recovered latent-normalization and response-design interfaces. No weights saved.

## Run 181
Run-180 NE gate with ridge 100 had held-out accuracy .99979, active miss rate .00171 and no inactive firing. This establishes bundle discrimination only, not dose decoding or causal effect. Recovering architecture and exact retained channel interfaces before testing an independent NE channel. No weights changed.

## Run 182
Architecture confirms the existing channel enters through z posterior and a reserved prelude attention head. Recovering gating/normalization algebra and NE decoder dictionaries to extend that head, rather than remove another head. No weights changed.

## Runs 183–184
Run-183 located raw_candidates87 (NE coefficient vectors length 577) and recovered latent normalization: signature plus .02*q*carrier. Its diagnostic stopped at a NumPy/Torch mismatch before changing any weights. Added a safe conversion and inspected exact wiring, retained dose perturbations and capture return structure. No weights changed.

## Run 185
Verified a rank-one latent channel and unused value dimensions in the existing reserved attention head. Tested gated NE innovation decoders on factual and imposed half/double-dose encodings. These tests concern faithful input transmission, not causal truth. No weights changed.

## Run 186
Run-185 NE decoders had near-unit imposed-dose slopes but poor held-out innovation R2. Constructed an UNSAVED dual channel using posterior rows 156/89, a second latent carrier and reserved-head value dimension 50; tested held-out prelude decoding and insulin preservation. CPU weights restored in finally; submitted edits unchanged. Factual fit and free rollouts, not channel decoding, will determine acceptance.

## Run 187
Run-186 preserved insulin posterior signal (RMS 1.5e-6) but NE prelude decoding R2 was .92 at amplitude 80. Tested amplitudes 10/20/40 with factual and half/double NE dose encodings, jointly fitting NE decoding and preservation of the current insulin B_in signal. Also computed MAP-selective response directions from the retained finite-difference Jacobian. CPU weights restored; no submitted weights changed. Record effect estimates remain conditional on the identification assumptions, not certified by these tests.

## Run 188
Run-187 amplitude 10/20/40 channels had held-out NE decoding R2 ~.976 and insulin-preservation RMSE ~.028, but dose increment slopes were only ~.794 (half) and .812 (double). Smaller amplitudes did not resolve attenuation. Checking physical-rate/index alignment and posterior versus E-decoder sensitivity before calibration. Recovered factual evaluation interfaces and replayed the submitted baseline. No saved weights changed.

## Run 189
Run-188 alignment checks ruled out indexing/rate mismatch. The run-180 gated decoder itself attenuates imposed NE dose changes, whereas the older decoder is nearly proportional on its smaller active subset. Recovering exact channel/gate construction before modifying it; classification accuracy alone does not establish faithful dose transmission. No saved weights changed.

## Run 190
Inspecting exact gate/channel interfaces and active gate margins over all retained dose augmentations. This distinguishes gate suppression from raw decoder approximation/extrapolation error before any edit. No saved weights changed.

## Run 191
The NE channel uses silu(g+q/2)-silu(g-q/2). Retrieving compact gate-margin results and cached fitting arrays after disassembly output truncation. No saved weights changed.

## Run 192
Run-191 gates were saturated on almost all held-out active bundles; raw NE decoder approximation is the main remaining transmission issue. Fitted closed-form decoder corrections, anchoring factual outputs and supervising imposed half/double physical-rate increments on training stays. Held-out transmission tests do not estimate drug effects. No saved weights changed.

## Run 193
Run-192 ridge .001/strength 100 improved raw held-out NE dose slopes to .989/.950 with factual signal RMSE .130. Constructed an UNSAVED amplitude-20 dual channel using this decoder and refitted joint NE decoding/insulin-preservation vectors on stay-disjoint training bundles. CPU weights restored to submitted edits in finally; no saved weights changed. Transmission validation is separate from record-based causal identification.

## Run 194
Run-193 amplitude-20 dual channel with ridge .0001 E decoding had NE R2 .980, insulin-preservation RMSE .02845 and held dose slopes .867/.929. Tested unsaved MAP-selective NE response scales 0/.5/1/1.5/2 with training factual replay; original/submitted baselines and per-event NLL deltas retained in results194. CPU restored; no saved weights changed. GPU factual and free-rollout validation are still required.

## Run 195
Saved provisional dual insulin/NE channel: amplitude 20, recalibrated NE decoder (.001,100), joint E ridge .0001, MAP-response direction ridge .1 at scale 1. Existing insulin response preserved by joint refit. Training replay dNLL +.0040994 versus original; scales 1.5/2 failed the budget. Previous submitted tensors backed up in backup_before195. GPU factual and rollout validation pending; causal readout targets remain record-based estimates subject to identification assumptions.


## Run 196
GPU factual validation on 98 tuning stays failed: original 1.59352, edited 1.60173, delta +.00821. Recovering architecture, evidence and retained candidate/replay interfaces to isolate the NE extension's damage before revising it. No weights changed.


## Run 197
Architecture and README recovered. Dose readout averages over stays, making equal-stay record estimates the relevant provisional targets (insulin x2 -13.2008, NE x2 +3.6087), not row-weighted estimates. Recovering compact candidate constructors and prior forward baselines after Run-196 output truncation. No weights changed.


## Run 198
Backed up failed dual candidate in backup_dual198. Inspecting response-scale CPU results, prior GPU rollouts and exact candidate/evaluation construction. No weights changed. Need to distinguish structural channel damage from response-strength damage on held-out stays.


## Run 199
Verified submitted B_in equals preserved insulin connection plus scale-1 NE rank-one response. Saved NE response scale .5, all channel and insulin weights unchanged. Training replay dNLL at .5 is +.0022796 versus original. Failed scale-1 dual weights remain backed up in backup_dual198. GPU factual and rollout checks pending; prior rollout CSVs summarized using eligible-hour, equal-stay aggregation.


## Run 200
Scale-.5 NE edit passed GPU factual (97 tuning stays, dNLL +.00233) but failed NE rollout signs (row means x2 -1.131, half +.797). Insulin row means x2 -10.463, half +12.317. Saved passing factual candidate in backup_passing200. Immediate NE Jacobian calibration is not sufficient for sustained rollout repair; investigating recurrent response/decoder feedback. Tool summaries explicitly report stay-hour means despite architecture's stay-averaging description, so retain both aggregates and matched targets. No weights changed.


## Run 201
Recovered architecture, record-based targets and exact dual-channel construction/evaluation interfaces to diagnose NE rollout sign reversal. No weights changed. Passing factual candidate remains backed up in backup_passing200.


## Run 202
Compared factual replay at NE connection scales +.5, 0, -.5 and -1, reconstructing the unit response from the scale-1 backup. Submitted weights unchanged; CPU globals restored. Reversed connection is a mechanism test, not evidence about the ICU's drug effect.


## Run 203
Provisionally saved NE B_in scale -.5, with every other channel/insulin weight unchanged. Training replay dNLL is approximately +.00447 versus original; scale -1 fails. Previous passing +.5 candidate saved in backup_before203 and backup_passing200. Matched tuning IDs recovered from fwd_17. GPU factual and rollout validation pending.


## Run 204
GPU reversed-NE candidate (scale -.5) passed factual on 98 tuning stays: 1.59803 versus original 1.59352, dNLL +.00451. Paired rollout row means: insulin x2 -8.493, half +12.367; NE x2 +1.345, half -4.699. This confirms rollout orientation reversal but not magnitude accuracy. Inspecting free baseline drift/bin saturation and nonlinear record evidence; no weights changed.


## Run 205
Compared recorded-dose generated baselines with matched actual tuning outcomes and inspected regularized response-direction alternatives. Mean generated values are not at global bin limits; large baseline shifts make recurrent feedback a plausible source of half/double asymmetry. No weights changed.

## Run 207
Run-206 stopped at a float32/float64 replay mismatch before returning factual scores. Recovering architecture, record-identification evidence, retained candidates and replay interfaces. No save_edit or drop_edit calls in this run; submitted weights are not changed.

## Run 208
Run-207 architecture review confirms recurrent and generated-reading feedback matters. Its long output was truncated; retrieving compact candidate/replay metadata to repair the dtype failure. No submitted weights changed.

## Run 209
Run-208 localized the failed comparison to an unsaved float64 B_in tensor. Replayed original, submitted and regularized NE-response candidates with all tensors float32. CPU mean losses: {'original': 1.6355364322662354, 'submitted': 1.6400082111358643, 'fixed206': 1.6379451751708984, 'ridge1_scale-0.5': 1.6388460397720337, 'ridge1_scale-1': 1.6412367820739746, 'ridge10_scale-0.5': 1.6381863355636597, 'ridge10_scale-1': 1.6389878988265991}. CPU globals restored in finally; no submitted weights changed. Factual loss screens candidates but does not establish rollout response or causal truth.

## Runs 210–211
Run-210 saved the ridge-10 NE connection at scale -1 (CPU training dNLL .003451), then failed while looking up a guessed rollout filename; its notebook write did not execute. Previous saved tensors remain in backup_before210. Locating actual forward artifacts now. No additional weights changed in Run-211; held-out validation remains outstanding.

## Run 212; forward passes used 20/50
Ridge-10 NE connection scale -1 passed GPU factual on 98 tuning stays: original 1.59352, edited 1.59702, delta .00350. Paired dose rollout row shifts: insulin x2 -10.385, half +12.633; NE x2 +.730, half -3.130. Both channels retain pronounced half/double asymmetry; inspecting causal evidence and dose transmission before changing strengths. No weights changed in this run.

## Run 213
Current ridge-10 rollout equal-stay shifts are insulin x2 -8.722, half +11.677; NE x2 +.609, half -2.840. Row-weighted shifts differ materially, so preserve both. Recovering cached dose-augmentation tests and record-effect estimates to distinguish input transmission from recurrent distortion. No weights changed.

## Run 214
Recovered architecture, record-identification evidence and retained repair interfaces. Current GPU-passing edit has strongly asymmetric half/double rollout effects; dose transmission and generated-state feedback must be distinguished before further calibration. No saved weights changed.

## Run 215
Retrieved compact repair interfaces, record-effect/kernel objects and dose-transmission diagnostics after Run-214 output truncation. Architecture confirms recurrent and generated-reading feedback; delayed record-derived insulin effects require longitudinal rather than immediate calibration. No saved weights changed.

## Run 216
Retrieved numerical longitudinal kernels, channel-construction variables and exact replay dependencies. Investigating whether doubled-dose saturation explains the half/double asymmetry. Record-derived targets remain provisional causal estimates. No saved weights changed.

## Run 217
Tested the submitted insulin/NE channels on held-out-context half/double action encodings, holding patient context fixed. Results retained in r217_transmission. This measures channel transmission, not causal drug effects. CPU globals restored in finally; no saved weights changed.

## Run 218
Run-217 fixed-context transmission slopes: insulin half .85884/double .87482; NE half .87314/double 1.01000. Similar insulin slopes and stronger NE doubling transmission do not explain the free-rollout asymmetry. Recovering impulse tests and temporal rollout profiles to distinguish hidden-state dynamics from generated-reading feedback. No saved weights changed.

## Run 219
Recovered architecture, retained impulse interfaces and latest hourly rollout profiles. Fixed-context dose transmission did not explain half/double asymmetry; investigating recurrent/generated-reading feedback before another weight edit. No saved weights changed.

## Run 220
Hourly GPU profiles show accumulating insulin effects and NE doubling with an early wrong sign. Recovering record-derived dynamics and candidate construction to distinguish delayed response from feedback distortion. No saved weights changed.

## Run 221
Run-220 output was truncated before late construction details. Retrieved targeted kernel and candidate metadata with bounded output. No saved weights changed; checking initial response and persistence against record-derived kernels.

## Run 222
Inspecting retained impulse results and factual replay implementation, with compact output. Considering longitudinal response-direction calibration rather than further immediate-response scaling. No saved weights changed.

## Run 223
Retained teacher-forced impulses decay rapidly after subsequent observed readings, unlike the multi-hour record-derived kernels. Inspecting isolated latent carriers for a factual-safe dose-dependent adjustment; generated-reading feedback remains a key limitation. No saved weights changed.

## Run 224
Comparing all retained free-rollout candidates using eligible-hour equal-stay readouts and inspecting paired-neuron gate geometry before nonlinear surgery. No saved weights changed; checking whether dual-channel construction introduced insulin asymmetry.

## Run 225
Screened UNSAVED insulin paired-gate nonlinear adjustments: compressed action gate, innovation thresholds at .5/1/1.5 factual carrier SD, response gains 2/3. These are computational correction candidates, not evidence of nonlinear ICU pharmacology. CPU factual results: {(0.5, 2.0): {'nll': 5.1639885902404785, 'delta': 3.528452157974243}, (1.0, 2.0): {'nll': 5.086690425872803, 'delta': 3.4511539936065674}, (1.5, 2.0): {'nll': 5.06265115737915, 'delta': 3.427114725112915}, (0.5, 3.0): {'nll': 3.610856294631958, 'delta': 1.9753198623657227}, (1.0, 3.0): {'nll': 3.6108880043029785, 'delta': 1.9753515720367432}, (1.5, 3.0): {'nll': 3.610907554626465, 'delta': 1.9753711223602295}}. No saved weights changed. GPU factual and dose-rollout validation required.


## Run 226
Rejected unsaved nonlinear paired-gate candidates from Run 225: factual losses increased by 1.98–3.53 nats. Recovering architecture, record-based targets and retained repair interfaces for a controlled, factual-safe adjustment. No saved weights changed.

## Runs 227–228
Run 227 inspection stopped on a display-only size/type error; no edits occurred. Retrieved retained longitudinal kernels, targets and local function interfaces to choose a specific calibration experiment. No saved weights changed.

## Run 229
Recovered record-derived tuning targets: insulin x2 about -7.8, half +3.9; NE x2 about +2.2, half -1.1 (subset/equal-stay aggregation). Current insulin half response is excessive. Inspecting innovation centering and response injection to test feedback distortion rather than blindly increasing gains. No saved weights changed.

## Run 230
Current channel uses policy-centered dose innovations. Testing whether generated-reading feedback or paired-neuron asymmetry explains excessive half-dose response. Later notebook targets differ by subset (insulin x2 -13.2008, NE x2 +3.6087); will recompute matched targets before calibration. No saved weights changed.

## Run 231
Paired-channel gating may explain Run-225 failure: inactive-bundle raw innovations can be large, so gate compression can leak. Inspecting factual and imposed-dose gate margins; considering only downward threshold shifts that preserve inactive suppression. No saved weights changed.

## Run 232
Screened UNSAVED insulin paired-gate downward bias shifts 16/20/23/26/30 with response gains 1/1.5. Unlike compression, downward shifts cannot activate previously suppressed non-insulin bundles. CPU factual results: {(16.0, 1.0): {'nll': 1.6390044689178467, 'delta': 0.003468036651611328}, (16.0, 1.5): {'nll': 1.7034400701522827, 'delta': 0.06790363788604736}, (20.0, 1.0): {'nll': 1.6389695405960083, 'delta': 0.0034331083297729492}, (20.0, 1.5): {'nll': 1.7033016681671143, 'delta': 0.0677652359008789}, (23.0, 1.0): {'nll': 1.6385912895202637, 'delta': 0.0030548572540283203}, (23.0, 1.5): {'nll': 1.7019946575164795, 'delta': 0.06645822525024414}, (26.0, 1.0): {'nll': 1.63834810256958, 'delta': 0.0028116703033447266}, (26.0, 1.5): {'nll': 1.696295976638794, 'delta': 0.060759544372558594}, (30.0, 1.0): {'nll': 1.639326810836792, 'delta': 0.0037903785705566406}, (30.0, 1.5): {'nll': 1.6956855058670044, 'delta': 0.06014907360076904}}. CPU globals restored in finally; no saved weights changed. GPU rollout needed to test asymmetry.

## Run 233
Saved only insulin paired-gate bias shift -26 (rows 96/184), leaving B_in and all other tensors at the previously GPU-passing candidate. CPU dNLL .002812. Full insulin-matrix gain scaling failed (~.06 nats), likely because it scales structural compensation too. Backup before this gate edit is backup_before233. Corrected action-mask inspection: active94 is not an insulin mask. GPU factual and rollout validation pending.

## Run 234; forward passes used 22/50
Threshold-26 insulin gate candidate passed GPU factual: original 1.59352,
edited 1.59741, delta .00389 on 98 tuning stays. GPU reached-row rollout
shifts: insulin x2 -8.773, half +6.696; NE x2 +.572, half -3.204.
Relative to the previous candidate (-10.385, +12.633, +.730, -3.130),
the gate shift markedly reduced excessive insulin half-dose response.
This supports a computational gating contribution, not nonlinear ICU
pharmacology. Recovering matched record-derived targets and construction
before further calibration. No weights changed in Run 234.

## Run 235
Architecture recovered; previous display truncated before retained objects.
Retrieving compact kernel/target/construction metadata. Current saved
threshold-26 candidate unchanged; GPU factual delta .00389.

## Run 236
Retrieving identification evidence and precise retained repair dependencies.
No weights changed. Longitudinal targets remain evidence-based estimates,
not simulator truth; aggregation must match the reached-row readout.

## Run 237
Compact numerical kernel and repair-interface extraction; no saved weights changed.

## Run 238
Applied retained 12-lag kernels to the exact fwd_23 reached-hour rows using
recorded doses from hour 12 onward. Checking raw-rate versus estimation
units before accepting matched targets. No saved weights changed.

## Run 239
Recovering architecture, dose-unit definitions, identification evidence, and retained repair interfaces before accepting Run-238 raw-rate targets. No weights changed. Row-versus-stay aggregation and kernel units remain explicit calibration uncertainties.

## Run 240
Run-239 architecture confirms recurrent and generated-reading feedback. Using bounded identification/interface inspection after truncated output; no weights changed.

## Run 241
Recovering longitudinal identification and channel construction in bounded output. Threshold and adjusted next-hour insulin estimates support a negative effect, but cumulative targets require lag-unit verification. No weights changed.

## Run 242
The 12-lag kernels agree with earlier longitudinal estimates; causal interpretation remains conditional on sequential exogeneity and transferable additive dynamics. Inspecting exact replay and response/compensation separation before calibration. No weights changed.

## Run 243
Computing both rollout aggregations and factual paired-neuron gate margins. Considering selective imposed-dose amplification rather than whole-matrix scaling, which previously damaged factual fit. No weights changed.

## Run 244
Screened UNSAVED insulin innovation-only gains with paired-gate thresholds (1,26), (1.25,28), (1.5,30), (1.5,34); B_in and structural compensation unchanged. CPU factual results: {(1.0, 26.0): {'nll': 1.63834810256958, 'delta': 0.0028116703033447266}, (1.25, 28.0): {'nll': 1.6389822959899902, 'delta': 0.003445863723754883}, (1.5, 30.0): {'nll': 1.6402311325073242, 'delta': 0.004694700241088867}, (1.5, 34.0): {'nll': 1.640176773071289, 'delta': 0.004640340805053711}}. CPU globals restored in finally. No saved weights changed. GPU factual and rollout validation remain necessary.

## Run 245
Screened UNSAVED stricter insulin gates at innovation gains 1.5/2. CPU factual results: {(1.5, 40.0): {'nll': 1.639259696006775, 'delta': 0.0037232637405395508}, (1.5, 50.0): {'nll': 1.6385431289672852, 'delta': 0.0030066967010498047}, (1.5, 60.0): {'nll': 1.6385431289672852, 'delta': 0.0030066967010498047}, (2.0, 40.0): {'nll': 1.6398146152496338, 'delta': 0.0042781829833984375}, (2.0, 50.0): {'nll': 1.6385431289672852, 'delta': 0.0030066967010498047}, (2.0, 60.0): {'nll': 1.6385431289672852, 'delta': 0.0030066967010498047}}. These are computational transmission/safety tests, not pharmacological nonlinearity estimates. CPU globals restored; submitted weights unchanged.

## Run 246
Saved insulin innovation-only gain 1.5 / total paired-gate shift 40 from Run-245 candidate. Only z_post.0.weight and z_post.0.bias changed relative to the prior submission; backup in r246_backup. CPU dNLL .0037233; GPU factual and rollout validation pending. Stricter 50/60 gates have identical factual scores and likely silence the factual channel; selected the less extreme threshold 40.

## GPU passes 23–24, after Run 246
Saved gain-1.5/gate-40 candidate passed factual validation on 98 tuning stays: original 1.59352, edited 1.59705, delta +.00353. Reached-row dose shifts were insulin x2 -2.364, half +.479, NE x2 +.879, half -3.240. The strict gate largely suppresses insulin transmission, rather than resolving its calibration. Recovering construction and identification units before selecting an intermediate gate. No new edits.

## Runs 247–248
Read architecture and record schema. Run-247 display truncated before candidate metadata; Run-248 requests bounded construction metadata. No saved weights changed. Gate-40 candidate remains GPU factual-safe on the tested subset but under-responsive for insulin.

## Run 249
Saved retained insulin innovation-only gain 1.25 / total gate shift 28 candidate from Run 244. Only z_post.0.weight/bias changed; checked all other edited tensors equal the candidate. Backup r249_backup preserves gate-40 submission. CPU screen delta +.00344586. GPU factual and rollout validation pending. This is a computational calibration experiment, not new evidence about ICU effects.

## Run 250; GPU passes 25–26
Gain-1.25/gate-28 candidate (hash 29031359da) passed GPU factual: 1.59668 vs 1.59352, delta +.00316 on 98 stays. Reached-row shifts: insulin x2 -7.539, half +3.566; NE x2 +.705, half -3.260. Insulin asymmetry is much reduced; NE calibration remains problematic. Saved passing backup r250_passing. Auditing longitudinal-kernel dose units and aggregation before accepting calibration targets. No weights changed.

## Run 252
Run 251 failed on MultiIndex column selection, not on estimation. Recovering architecture, retained interfaces, and raw/panel dose units. No weights changed; gain-1.25/gate-28 remains the saved candidate.

## Run 253
Run 252 architecture inspection succeeded, but output truncation hid the unit audit. Bounded inspection of dose normalization, calibration names, and early identification notes. No weights changed.

## Run 254
Raw-rate panel units confirmed (insulin 0–6, NE 0–1); NE recorded total dose is separately normalized. Earlier RD and history-adjusted estimates support insulin negative/NE positive effects, subject to identification assumptions. Recovering lag kernels and response-channel interfaces. No weights changed.

## Run 255
Recovered 12-lag raw-rate kernels: insulin approximately [-2.05,-2.48,-2.21,-1.80,-1.37,-1.01,-.75,-.58,-.42,-.29,-.25,-.08]; NE [3.14,2.56,1.49,.77,.47,.38,.23,-.008,.22,.087,.019,.064]. Checking exact dose fixtures and reached-hour output schema before matched calibration. No weights changed.

## Run 256
Training identification dose fixtures match raw panel rates exactly. Computed exact reached-hour convolution targets and both row/stay aggregation from latest GPU output. Results: {"insulin_x2": {"hours": 507, "stays": 51, "model_row_mean": -7.539431163708087, "model_stay_mean": -6.809219835469541, "target_row_mean": -15.009001095986699, "target_stay_mean": -13.193302494613517, "kernel_bootstrap_95pct": [-14.727656386918879, -11.778734603946658]}, "insulin_half": {"hours": 507, "stays": 51, "model_row_mean": 3.566270611439842, "model_stay_mean": 3.0032466817333, "target_row_mean": 7.5045005479933495, "target_stay_mean": 6.596651247306759, "kernel_bootstrap_95pct": [5.889367301973329, 7.3638281934594385]}, "ne_x2": {"hours": 322, "stays": 29, "model_row_mean": 0.7052978260869566, "model_stay_mean": 0.5986483411703241, "target_row_mean": 4.865639862977985, "target_stay_mean": 4.70503970884939, "kernel_bootstrap_95pct": [4.391951765461908, 5.153332151030425]}, "ne_half": {"hours": 322, "stays": 29, "model_row_mean": -3.259521739130435, "model_stay_mean": -2.9594237617554864, "target_row_mean": -2.4328199314889924, "target_stay_mean": -2.352519854424695, "kernel_bootstrap_95pct": [-2.576666075515213, -2.195975882730954]}}
Bootstrap intervals quantify kernel estimation only, not identification or extrapolation bias. No weights changed.

## Runs 257–258
Run 257 aborted before evaluation on NumPy/Torch subtraction; no edits. Run 258 preserves tensor types and screens the same fixed-direction insulin candidates. Results: {(1.25, 28.0): {'nll': 1.6389822959899902, 'delta': 0.003445863723754883}, (1.5, 28.0): {'nll': 1.6392815113067627, 'delta': 0.0037450790405273438}, (1.75, 30.0): {'nll': 1.6407108306884766, 'delta': 0.005174398422241211}, (2.0, 32.0): {'nll': 1.6411798000335693, 'delta': 0.005643367767333984}, (2.25, 34.0): {'nll': 1.6421180963516235, 'delta': 0.006581664085388184}, (2.5, 36.0): {'nll': 1.641078233718872, 'delta': 0.005541801452636719}, (3.0, 40.0): {'nll': 4.2238993644714355, 'delta': 2.5883629322052}}
CPU globals restored; saved weights unchanged. GPU rollout and factual validation remain necessary.

## Run 259
Saved insulin innovation gain 1.5 / gate shift 28. Only z_post.0.weight/bias changed from r250_passing; backup r259_backup. CPU delta +.00374508. Stronger screened settings mostly exceed .005. GPU factual and dose rollout validation pending. Inspecting NE posterior rows 156/89 for an analogous selective calibration.

## GPU passes 27–28; Run 260
Gain-1.5/gate-28 passed GPU factual on 98 stays: delta +.00327. Reached-row rollout shifts: insulin x2 -7.777, half +2.207; NE x2 +.748, half -3.178. Increasing insulin gain from 1.25 did not proportionally increase response and worsened halving. Inspecting retained response/compensation construction before further edits.

## Run 261
Bounded recovery of posterior pair construction, candidate helpers, and factual replay. No saved edits changed.

## Run 262
Recovered sparse edit support and rollout schema; reviewing earlier calibration history. Saved weights unchanged.

## Run 263
Historical GPU dose-shift comparison and recovery of factual scoring interfaces. One-sided insulin pair and gate suppression motivate a controlled gate screen. No weights saved.

## Run 264
UNSAVED lower-gate insulin factual screen: {(1.0, 22.0): [{'shape': (4159, 992), 'mean': 0.10410576313734055}, {'shape': (4699,), 'mean': 1.6389542818069458}, {'shape': (4699, 17), 'mean': 0.05882352963089943}], (1.25, 24.0): [{'shape': (4159, 992), 'mean': 0.10685418546199799}, {'shape': (4699,), 'mean': 1.6387673616409302}, {'shape': (4699, 17), 'mean': 0.05882352963089943}], (1.5, 24.0): [{'shape': (4159, 992), 'mean': 0.10894965380430222}, {'shape': (4699,), 'mean': 1.6396914720535278}, {'shape': (4699, 17), 'mean': 0.05882352963089943}], (1.5, 26.0): [{'shape': (4159, 992), 'mean': 0.1134854108095169}, {'shape': (4699,), 'mean': 1.6398966312408447}, {'shape': (4699, 17), 'mean': 0.05882352963089943}], (1.75, 26.0): [{'shape': (4159, 992), 'mean': 0.11500158905982971}, {'shape': (4699,), 'mean': 1.641042709350586}, {'shape': (4699, 17), 'mean': 0.05882352963089943}], (2.0, 28.0): [{'shape': (4159, 992), 'mean': 0.11917470395565033}, {'shape': (4699,), 'mean': 1.6406002044677734}, {'shape': (4699, 17), 'mean': 0.05882352963089943}]}. CPU globals restored; saved weights unchanged. Reviewing recurrent NE response construction.

## Run 265
UNSAVED symmetric insulin-pair factual screen: {(1.25, 22.0): {'nll': 1.6400622129440308, 'delta': 0.00452578067779541}, (1.5, 22.0): {'nll': 1.641606092453003, 'delta': 0.006069660186767578}, (1.5, 26.0): {'nll': 1.6392319202423096, 'delta': 0.0036954879760742188}, (2.0, 22.0): {'nll': 1.6457823514938354, 'delta': 0.010245919227600098}, (2.0, 26.0): {'nll': 1.6408699750900269, 'delta': 0.005333542823791504}, (2.5, 26.0): {'nll': 1.6431478261947632, 'delta': 0.007611393928527832}}. Pair uses shared action gate with +/- half the innovation, and exact opposite output columns. CPU globals restored; submitted tensors unchanged.

## Run 266
Provisionally saved symmetric insulin pair, gain 1.5/gate 26, changing only z_post.0.weight/bias. CPU dNLL +.00369549. Prior gain-1.5/gate-28 one-sided candidate backed up in backup266. GPU rollout and factual validation pending.

## Run 267
Recovered architecture, tuning IDs, latest forward schemas and retained helpers before GPU validation of Run-266 symmetric insulin candidate. No weights changed. CPU factual delta remains +.00369549; GPU factual fit and rollout response are unvalidated.

## Run 268
Recovered exact prior GPU validation stay IDs and bounded calibration helper metadata. No weights changed. Run-266 symmetric pair awaits paired GPU factual and dose-edit validation.

## Runs 269–270; GPU passes 29–30
Run 269 halted at a pandas shift-column/method collision after creating backup269; no weights changed. Run 270 corrects reporting and inspects calibration helper construction. Symmetric insulin gain 1.5/gate 26 passed GPU factual with delta +.00401. Reached-row shifts: insulin x2 -10.009, half +8.278; NE x2 +.536, half -3.324. Shared-gate NE repair is a computational hypothesis, not new evidence of ICU effects.

## Run 271
Checked NE rows 156/89 and screened unsaved shared-gate, symmetric innovation candidates where output-column geometry permits. Antipodal error 0.362748; CPU results {}. All model-weight dictionaries restored; submitted weights unchanged. These tests address model transmission, not ICU-effect identification.

## Runs 272–273
Run 272 failed because factual_eval194 requires a complete tensor dictionary, not the ten-tensor submission backup; globals restored and no saved weights changed. Run 273 supplied all 181 tensors. CPU control 1.6392319202423096; candidate results {(0.8, False): {'nll': 1.6389565467834473, 'delta': 0.003420114517211914}, (1.0, False): {'nll': 1.6392323970794678, 'delta': 0.003695964813232422}, (0.8, True): {'nll': 1.6389565467834473, 'delta': 0.003420114517211914}, (1.0, True): {'nll': 1.6392325162887573, 'delta': 0.0036960840225219727}}. NE symmetry candidates remain unsaved pending selection and GPU tests.

## Run 274
Recovered architecture and recent NE candidate construction metadata. Run-273 CPU screening alone does not establish rollout improvement. No submitted weights changed.

## Run 275
Provisionally saved shared-gate symmetric NE gain 1.0 with exact opposing output columns, retaining insulin gain 1.5/gate 26. CPU screen dNLL +.00369608. Changes: {'z_post.0.bias': 0.5078014135360718, 'z_post.0.weight': 142.60923767089844, 'z_post.2.weight': 0.2676766514778137}. Previous submission backed up in backup275. GPU rollout and factual validation pending.

## GPU pass 31; Runs 276–277
Run 276 restored backup275 before a diagnostic type error; its log append was not reached. Symmetric NE gate 0 had rollout shifts insulin -11.655/+9.265, NE +.072/-.513 and was rejected. Run 277 screened unsaved gate openings with exact opposing NE columns: {0.0: {'nll': 1.6392323970794678, 'delta': 0.003695964813232422}, -10.0: {'nll': 1.6392323970794678, 'delta': 0.003695964813232422}, -20.0: {'nll': 1.6392326354980469, 'delta': 0.0036962032318115234}, -40.0: {'nll': 1.6427125930786133, 'delta': 0.00717616081237793}, -80.0: {'nll': 5.6031060218811035, 'delta': 3.967569589614868}, -120.0: {'nll': 5.603105545043945, 'delta': 3.96756911277771}, -200.0: {'nll': 5.6031060218811035, 'delta': 3.967569589614868}}. Model dictionaries restored; submitted weights remain backup275.

## Run 278
Unsaved NE gain/output-pair screen: {(1.0, False, 0.0): {'nll': 1.6392323970794678, 'delta': 0.003695964813232422}, (2.0, False, 0.0): {'nll': 1.6416304111480713, 'delta': 0.0060939788818359375}, (4.0, False, 0.0): {'nll': 1.6503831148147583, 'delta': 0.01484668254852295}, (8.0, False, 0.0): {'nll': 6.819105625152588, 'delta': 5.1835691928863525}, (2.0, True, 0.0): {'nll': 1.6416305303573608, 'delta': 0.006094098091125488}, (4.0, True, 0.0): {'nll': 1.6503831148147583, 'delta': 0.01484668254852295}, (4.0, False, -20.0): {'nll': 3.0426669120788574, 'delta': 1.407130479812622}}. Tested transmission hypotheses only; ICU effect estimates unchanged. Global model dictionaries restored; no submitted weights changed.

## Run 279
Unsaved asymmetric insulin innovation screen (increase/decrease gains), including modest NE gain: {(2.0, 1.0, 26.0, None): {'nll': 1.639370322227478, 'delta': 0.0038338899612426758}, (2.25, 0.75, 26.0, None): {'nll': 1.6394871473312378, 'delta': 0.003950715065002441}, (2.5, 0.5, 26.0, None): {'nll': 1.639622449874878, 'delta': 0.004086017608642578}, (2.0, 1.0, 24.0, None): {'nll': 1.639768362045288, 'delta': 0.004231929779052734}, (2.25, 0.75, 28.0, None): {'nll': 1.6395223140716553, 'delta': 0.003985881805419922}, (2.25, 0.75, 26.0, 1.4): {'nll': 1.6402515172958374, 'delta': 0.004715085029602051}}. All model dictionaries restored; submitted tensors unchanged. Asymmetry is a rollout calibration hypothesis, not new ICU-effect evidence.

## Run 280
Provisionally saved asymmetric insulin pair, increase gain 2.25 / decrease gain .75 / gate 26, retaining pre-275 NE construction. CPU dNLL +.00395072; GPU validation pending. Backup backup280. Recovering action fixture helpers to test robust NE identity gating.

## Run 281
Recovered architecture, record-based identification notes, latest validation IDs and helper signatures. No weights changed. Run-280 asymmetric insulin candidate still needs GPU factual and rollout validation.

## Run 282
Bounded recovery of target-estimation notes, validation stay IDs and calibration helper names after Run-281 output truncation. No weights changed.

## GPU passes 32–33; Run 283
Asymmetric insulin 2.25/.75 gate 26 predicts reached-row shifts -11.957/+9.592; NE +.620/-3.188. GPU factual delta +.00453 on 98 tuning stays: passes narrowly. Halving did not decrease as hypothesized. Recovering longitudinal targets and NE gating before further edits; no weights changed.

## Run 284
Recovered exact matched record-kernel targets and paired-neuron construction to select a specific calibration. No weights changed. Targets remain estimates under the longitudinal identification assumptions, not supplied simulator truth.

## Run 285
Fitted a small NE identity classifier on exact action embeddings spanning rates 0–4, insulin 0–32, mixed actions and no action; no model gradients. Built unsaved shared-gate opposing NE pair and screened CPU factual loss: {(0.5, 40.0): {'nll': 1.640046238899231, 'delta': 0.0045098066329956055}, (0.8, 40.0): {'nll': 1.6418455839157104, 'delta': 0.006309151649475098}, (1.0, 40.0): {'nll': 1.6435017585754395, 'delta': 0.007965326309204102}, (1.0, 80.0): {'nll': 1.643504023551941, 'delta': 0.007967591285705566}}. Model dictionaries restored; submitted weights unchanged. Matched record targets (row means): insulin -15.009/+7.505, NE +4.866/-2.433; equal-stay targets -13.193/+6.597 and +4.705/-2.353.

## Run 286
Refit NE identity gate over empirical training dose ranges extended to half/double, including mixed bundles. CPU screening: {(2.25, 0.75, 27.0, 0.5): {'nll': 1.6390372514724731, 'delta': 0.003500819206237793}, (2.25, 0.75, 28.0, 0.5): {'nll': 1.6396654844284058, 'delta': 0.00412905216217041}, (2.0, 1.0, 28.0, 0.5): {'nll': 1.6397873163223267, 'delta': 0.004250884056091309}, (1.5, 1.5, 26.0, 0.5): {'nll': 1.63921058177948, 'delta': 0.003674149513244629}, (2.25, 0.75, 28.0, 0.7): {'nll': 1.6402779817581177, 'delta': 0.004741549491882324}}. Provisionally selected (2.25, 0.75, 27.0, 0.5); prior edit backed up in backup286. GPU rollout/factual validation pending. This is a transmission calibration, not new causal evidence.

## Run 287
Recovered architecture, saved-edit metadata, and latest GPU validation files. No weights changed. Run-286 candidate still requires GPU rollout and factual checks.

## Run 288
Recovered exact GPU validation IDs and output schemas after truncated recovery. No weights changed. Run-286 candidate awaits paired GPU validation.

## GPU pass 34; Run 289
Run-286 identity gate failed rollout: insulin -8.590/+7.026; NE -1.604/+4.766. Restored backup286, the prior GPU-validated candidate. Screened unsaved reversed NE transmission variants: {(2.25, 0.75, 26.0, -0.3): {'nll': 1.6389594078063965, 'delta': 0.0013927221298217773}, (2.25, 0.75, 26.0, -0.5): {'nll': 1.6392511129379272, 'delta': 0.001684427261352539}, (2.5, 0.5, 26.0, -0.4): {'nll': 1.639230489730835, 'delta': 0.001663804054260254}, (3.0, 0.0, 26.0, -0.4): {'nll': 1.6395153999328613, 'delta': 0.001948714256286621}, (3.0, 0.0, 27.0, -0.4): {'nll': 1.6386622190475464, 'delta': 0.0010955333709716797}, (2.25, 0.75, 26.0, 0.0): {'nll': 1.638817310333252, 'delta': 0.001250624656677246}}. CPU globals restored. These are computational tests, not new evidence of ICU effects.

## Run 290
Corrected Run-289 reference: baseline_losses194 was an intermediate-model loss array, not the original. True same-fixture original CPU NLL = 1.6355364322662354. Corrected prior deltas = {(2.25, 0.75, 26.0, -0.3): 0.003422975540161133, (2.25, 0.75, 26.0, -0.5): 0.0037146806716918945, (2.5, 0.5, 26.0, -0.4): 0.0036940574645996094, (3.0, 0.0, 26.0, -0.4): 0.0039789676666259766, (3.0, 0.0, 27.0, -0.4): 0.003125786781311035, (2.25, 0.75, 26.0, 0.0): 0.0032808780670166016}. New screen = {(3.5, 0.0, 26.0, -0.4): {'nll': 1.6406742334365845, 'delta': 0.005137801170349121}, (4.0, 0.0, 27.0, -0.4): {'nll': 1.6406702995300293, 'delta': 0.005133867263793945}, (4.0, 0.0, 26.0, -0.4): {'nll': 1.6420460939407349, 'delta': 0.006509661674499512}, (3.0, 0.0, 26.0, -0.4): {'nll': 1.6395153999328613, 'delta': 0.0039789676666259766}}. Provisionally saved (3.0, 0.0, 26.0, -0.4); GPU factual and rollout validation pending.

## GPU passes 35–36; Run 291
Run-290 candidate rollout: insulin -15.571/+9.247, NE -1.128/-4.164. GPU factual delta +.00364 on 98 stays. Backed up as backup291. Screened earlier NE construction: {(3.0, 0.0, 26.0, 1.0): {'nll': 1.6398972272872925, 'delta': 0.004360795021057129}, (3.0, 0.0, 26.0, 1.1): {'nll': 1.6400619745254517, 'delta': 0.004525542259216309}, (3.0, 0.0, 26.0, 1.4): {'nll': 1.6406618356704712, 'delta': 0.00512540340423584}, (3.0, 0.0, 27.0, 1.4): {'nll': 1.6398664712905884, 'delta': 0.004330039024353027}}. Provisionally saved (3.0, 0.0, 26.0, 1.0); rollout unvalidated. Record-based effect estimates unchanged.

## Run 292
Recovered architecture, recent forward schemas/cohort, edit helper names, and record-identification notes. No weights changed. Current (3.0, 0.0, 26.0, earlier-NE gain 1.0) candidate awaits GPU validation.

## GPU pass 37; Run 293
Current earlier-NE candidate (insulin 3/0, gate 26, NE gain 1) predicts reached-row shifts -17.143/+10.117, NE +0.075/-0.491 on 98 tuning stays. Insulin overshoots the matched record estimates; NE transmission remains ineffective. Recovered helpers and backups; no weights changed.

## Run 294
Saved asymmetric insulin 3/0, gate 27, earlier-NE gain 1.4. Prior same-fixture CPU dNLL +0.00433004. Run-293 submission backed up as backup294. Hypothesis: reduce insulin overshoot while strengthening NE; GPU rollout and factual checks pending. No new causal estimates.

## GPU passes 38–39; Run 295
Gate-27 earlier-NE gain 1.4 rollout -15.003/+7.246, NE -0.625/+0.464. Factual GPU penalty +0.00556 on 98 stays: rejected. Screened weaker/reversed NE injections against fresh original CPU NLL 1.63553655: {(3.0, 0.0, 27.0, 0.5): {'nll': 1.6385393142700195, 'delta': 0.003002762794494629}, (3.0, 0.0, 27.0, -0.5): {'nll': 1.6387571096420288, 'delta': 0.0032205581665039062}, (3.0, 0.0, 27.0, -1.0): {'nll': 1.639501690864563, 'delta': 0.003965139389038086}, (3.0, 0.0, 27.0, 1.0): {'nll': 1.6391000747680664, 'delta': 0.003563523292541504}}. Saved (3.0, 0.0, 27.0, 0.5); GPU validation pending. No new causal evidence; targets remain record-derived estimates.

## Run 296
Recovered architecture, current editing helpers, record-identification notes and latest GPU output schemas after restart. No weights changed. Current insulin 3/0, gate 27, earlier-NE gain 0.5 candidate awaits GPU validation.

## GPU passes 40–41; Run 297
Prior insulin 3/0, gate 27, earlier-NE +0.5 candidate failed factual fit on tuning IDs 2020000000–2020000099: original NLL 1.59524, edited 1.60153, penalty +0.00629. Its reached-row rollout shifts were insulin -16.221/+6.791, NE -3.644/+1.880; original -0.382/+0.209, NE +0.103/-0.307. Backed up as backup297. Saved provisional insulin 2.75/0, gate 28, earlier-NE -0.5. Hypothesis: weaker insulin transmission improves factual fit and reversed NE transmission corrects the sign. GPU validation pending. Record-derived targets unchanged.

## GPU passes 42–43; Run 298
Candidate (2.75,0,28,-0.5) passed first-100 tuning factual fit: original 1.59524, edited 1.59802, penalty +0.00278. Its rollout shifts were insulin -4.859/+2.825, NE +5.016/-4.777. NE doubling is close to the record-derived target; NE halving overshoots, and insulin is too weak. Actual changed tensors audited; unchanged saved copies removed. Backed up as backup298. Saved provisional candidate (3,0,27,-0.5), strengthening insulin while retaining reversed NE. GPU factual and rollout validation pending; no new causal evidence.


## GPU passes 44–46; Run 299
Validated insulin 3/0, gate 27, earlier-NE -0.5 (tool hash c198f94cd7). First-100 tuning factual penalty +0.00319; second-100 +0.00473; reading-weighted pooled penalty +0.003961. First-100 reached-row shifts: insulin -11.158/+5.855, NE +5.001/-4.793. Preserved all current edited tensors in backup299. Used labeled saved snapshots to isolate an NE-gain direction and audit the insulin-gate direction; no gradients or optimiser. Gate audit passed=False; saved provisional candidate NE gain -0.45 only; gate reconstruction rejected. Hypothesis: modest gate reduction strengthens insulin, while weaker reversed NE transmission reduces halving overshoot. GPU factual and rollout checks pending. Record-derived effect estimates remain estimates under identification assumptions; these calibration trials provide no new evidence of simulator truth.


## GPU passes 47–48; Run 300 (last Python run)
NE-only interpolation -0.45, gate unchanged at 27, passed first-100 GPU factual with exact CSV penalty +0.0032205. Reached-row readouts: {'insulin_half': 6.729460583941606, 'insulin_x2': -11.03206806569343, 'ne_half': -3.9622215297450425, 'ne_x2': 3.714053824362606}. Compared with NE -0.5, both halving estimates improve, NE doubling weakens, and the main insulin doubling effect remains too small. Recovered a clean two-bias insulin gate direction; discarded all weight cancellation residuals (max 0.00012207031). Considered only a prespecified bias-only gate shift of -0.35, with no gradients or optimiser. Accepted it only if an existing CPU factual evaluator reproduced both documented original/current fixture losses and retained a conservative factual-loss margin. CPU check: None. Final selection: validated insulin 3/0, gate 27, NE -0.45. Actual changed tensors: ['core.b_in.weight', 'core.prelude.0.attn.kv.weight', 'core.prelude.0.attn.out.weight', 'core.prelude.0.attn.q.weight', 'z_in.bias', 'z_in.weight', 'z_post.0.bias', 'z_post.0.weight', 'z_post.2.bias', 'z_post.2.weight']. Weight SHA256: 943ca19d75f6e698e6591d285b1f1e92162bf2050cf5507f1d0d7b804b87bb35. Record-derived matched targets from Run 256 remain approximately insulin -15.009/+7.505 and NE +4.866/-2.433; these are estimated longitudinal effects, not known simulator truth. Final GPU factual confirmation is planned in /work/fwd_50/; final rollout confirmation in /work/fwd_51/. Their CSVs are the authoritative final measurements and supersede any expectation from this notebook. No Python runs remain.
