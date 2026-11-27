# Weight repair notebook

## Objective and constraints
Infer drug effects only from simulated ICU records; edit weights analytically without gradients or a weight optimizer. Preserve factual value NLL within +0.005 nats and examine all-output loss where available.

## Initial plan
Read architecture and data schema; identify sources of dose variation and confounding; estimate and falsify causal responses separately for both drugs; inspect model dose-sensitive states; construct a narrowly targeted, dose-generalizing edit; validate factual predictions and paired rollouts.

No drug effect has yet been assumed or estimated.

## Run 1
Read full architecture and schema. Action embeddings feed a shared recurrent state; last-value registers also feed value heads. No effect estimates or edits yet. Initial printed output was truncated; run 2 targets missing details.

## Run 2
Training has 30,000 stays and 1,997,441 events. Measurements at integer hours; actions at hour+0.0833, lasting one hour. Insulin rates are quantized in 0.5 steps; norepinephrine rates in 0.01 steps and capped at 1. No duplicate code/stay/hour rows. Constructed hourly panel and began raw/history/within-stay comparisons.

## Baseline rollout
First100 tuning original shifts: I x2 -3.680, I half +2.018, NE x2 +0.465, NE half -0.871. Rich regression results retained in rich_fits; they are diagnostic, not yet accepted causal estimates.

## Strengthened record evidence (run 4)
Nonlinear history controls restricted to active infusions give insulin next-hour coefficients around -2.0 (finger and lab agree); NE around +3.1 on arterial and +3.6 on cuff when both current sensors observed. Earlier near-zero estimates were misspecified, not evidence of absent effects. Original factual tuning first100 NLL = 1.63164, n=4751. Run 5 fits a separate two-sensor random-baseline state-space model to estimate persistence; this does not modify world-model weights.
glucose SSM parameters [144.1680625841136, 0.9263030096279322, 0.3154693959901756, 3.5763416283774636, 3.2934889974132835, 1.678039410652465, 2.1832857530220755, 1.1568945707731426, 0.037711265958233535], success True.
MAP SSM parameters [75.25256805642357, 0.8879818156248919, -0.3778968848036003, 2.512353157927126, 2.005848208979782, 0.6511878302787533, 1.167060058781382, 1.7996778798728403, 3.091933533327739], success True.
Provisional linear rollout targets: {"insulin_x2": 2.3193790074377105, "insulin_half": -1.1596895037188553, "ne_x2": -0.9474186067740055, "ne_half": 0.47370930338700273}

## Restart / run 6
SSM and rich-history coefficients disagree in sign. Treat SSM rollout targets as unvalidated, not causal truth. Recovering architecture, timing, and persistent analyses before choosing an edit.

## Run 7
Built independently aligned P (readings at integer hour) and A (drug records floored to their start hour). Inspecting treatment support and previous fit implementation; no causal target or edit accepted yet.

## Run 8
Insulin has sharp recorded-glucose dosing thresholds. Computing local-linear discontinuity ratios at 150.5,179.5,219.5,259.5, two bandwidths, both next-hour sensors, and pre-dose lab/observation-placebo checks. Also testing which MAP sensor drives NE selection. Results remain subject to placebo and precision checks.

## Run 9
Threshold evidence: insulin at 179.5 gives beta -2.04 (cluster SE .26, width12); at219.5 -2.28 (.48). Pre-dose lab jumps near zero; first150.5 threshold changes monitoring, so do not use that finger estimate alone. Implemented independent SSM with explicit observations -1..24 and action a[h] only in transition h->h+1, to test whether the old opposite-sign estimates arose from time alignment. No edits.

## Run 10
Explicitly aligned random-baseline SSM still conflicts with threshold evidence (glucose rho at .9999, beta -.033; MAP beta -.377). Therefore timing alone does not explain it. Fitting a separate profiled fixed-effect SSM with unrestricted stay-specific intercept and initial state; random-effect independence from adaptively selected treatments is suspect. This is a small record model, not world-model training.

## Run 11
Profiled fixed-effect SSM also gives incompatible effects (I -.419,rho .772; NE .435,rho .602), so neither simple SSM is accepted as a causal model. Insulin dose/base SD .343 overall and .330 within stay provides substantial within-stay variation. Built pre-action history features only and ran two-fold stay-separated flexible residualization (small statistical regressions, not world-model training), comparing all/active/both-current subsets and both sensors.

## Run 12 / second restart
Recovered architecture and persistent-variable inventory after original state export. No edits; insulin threshold evidence remains provisional, and neither conflicting SSM is accepted as causal truth. Next: inspect flexible residualization results and state/head pathways.

## Run 14
Recomputed cross-fitted two-treatment residual regressions with stay-clustered uncertainty and observation/active-treatment subsets; retained results in effects14. Inspecting NE policy support before accepting its effect.

## Run 15 / alignment audit
Run 14 subset masks mistakenly used columns 1:25 although P/A hour labels are 0:25. All-row residual coefficients were unaffected. Audited cross-fitted target reconstruction against offsets and recomputed masks with current hours 0:24; results retained in audit15 and effects15. No edits.

## Run 16
Alignment audit reconstructed dose at h and reading at h+1 exactly. Corrected DML: finger I=-1.936(.045), lab=-1.968(.079), arterial NE=3.049(.088), cuff=2.529(.172); arterial active/both-current=3.150(.158), cuff active/both-current=3.297(.428). Running independent insulin/NE threshold and pre-dose/monitoring-placebo checks. No edits.

## Run 17
Independent RD confirms finger insulin effect around -2 with pre-dose lab and monitoring placebos near zero. NE has no strong threshold first stage, so its RD ratios are NOT evidence. Estimated persistence using cross-fitted current-dose innovations, future outcomes, and future-dose residuals; models are separate record regressions, not world-model training. Results: {'glucose_finger': {'beta_anchor': -1.95, 'rho_h2': 1.1843901657922165, 'rho_h2_SE': 0.026718364441588156, 'kernel_lag2': -2.005241875550968, 'lag2_over_beta': 1.0283291669492143, 'rho_squared': 1.402780064825314}, 'glucose_lab': {'beta_anchor': -1.95, 'rho_h2': 1.1479572271952263, 'rho_h2_SE': 0.04674336033365004, 'kernel_lag2': -2.0225015960408346, 'lag2_over_beta': 1.0371803056619664, 'rho_squared': 1.3178057954697524}, 'map_arterial': {'beta_anchor': 3.1, 'rho_h2': 0.798530224335335, 'rho_h2_SE': 0.033821931868103325, 'kernel_lag2': 1.6581693284572787, 'lag2_over_beta': 0.5348933317604124, 'rho_squared': 0.6376505191770405}, 'map_cuff': {'beta_anchor': 3.1, 'rho_h2': 0.6343387617013322, 'rho_h2_SE': 0.059850520019256566, 'kernel_lag2': 1.3711487169076348, 'lag2_over_beta': 0.4423060377121403, 'rho_squared': 0.4023856645967795}}

## Run 18
Innovation-based persistence is not accepted as AR(1): I rho_h2=1.18 finger/1.15 lab but lag2/beta about1.03, inconsistent with rho^2; NE rho_h2=.80 arterial/.63 cuff, lag2/beta .535/.442. Running independent dynamic insulin RD and recovering exact head/vocab details. Dynamic RD kernels: {(179.5, 6, 'glucose_finger'): [-2.0, -2.2825629333444692, -1.6380767005368395, -0.6685756287878758, -0.43570675678634363], (179.5, 6, 'glucose_lab'): [-2.0, -2.6244192897242735, -1.323730803946258, -1.4465038671483337, -2.1328354352011347], (179.5, 12, 'glucose_finger'): [-2.0, -1.9286691838699255, -1.6655095667832636, -0.9973050809961331, -0.6217251413445759], (179.5, 12, 'glucose_lab'): [-2.0, -2.3843194891872472, -1.3381192860764797, -2.1112451007967996, -2.196939803863609], (219.5, 6, 'glucose_finger'): [-2.0, -2.439939771849313, -2.022240566690555, -1.8954732634405942, -1.6554350153186228], (219.5, 6, 'glucose_lab'): [-2.0, -4.207667644651504, -0.3301466973805788, 0.39552693719183585, -0.4555459557852317], (219.5, 12, 'glucose_finger'): [-2.0, -2.541688071630246, -1.9149790440306946, -2.185962256105463, -2.0262692223068313], (219.5, 12, 'glucose_lab'): [-2.0, -2.833701932078345, -2.2939384626089256, -1.5662332544028423, -1.9974355933520223]}

## Run 19
Independent dynamic RD shows delayed and declining insulin response, suggesting testing AR2 rather than forcing AR1. Higher-horizon innovation kernels and out-of-horizon AR2 predictions retained in kernels19 and ar2_predictions19. Measured kernels: {'glucose_finger': [-1.95, -2.309560823294822, -2.005241875550968, -1.603850386577875, -1.0841092995363333, -0.7821710394651412, -0.5042097400050339, -0.3755434211764299], 'glucose_lab': [-1.95, -2.2385165930306914, -2.0225015960408346, -1.4554437429171376, -1.1695181660533853, -0.8137586123330646, -0.5099624860342239, -0.2911248267002852], 'map_arterial': [3.1, 2.4754436954395387, 1.6581693284572787, 0.9433639872491649, 0.35728376967602093, 0.2745472164579554, 0.01817551193633286, 0.1668406960610107], 'map_cuff': [3.1, 1.9664501612741299, 1.3711487169076348, 0.5140990885791327, 0.6131312288760626, -0.17057540601591642, -0.06638572299454995, 0.07700460977812486]}

## Run 20 / restart recovery
Cross-sensor impulse responses support a delayed insulin effect and faster-decaying NE effect. AR1 is not accepted; inspecting the exact architecture and retained analysis objects before any weight edit. No tensors edited; forward passes remain 3/50.

## Run 21
Architecture: actions have a separate DeepSets encoder but enter shared posterior/state pathways. Values are a shared 192-hidden-unit bin head plus code/last-bin transition logits. Inspecting state-export schema and exact tensors for a low-rank, factual-preserving intervention correction. No edits.

## Runs 22–23
Run 22 stopped on a pandas shift-column/method collision; corrected in run 23. Provisional record-kernel targets: {'insulin_x2': np.float64(-10.675036415440115), 'insulin_half': np.float64(5.337518207720057), 'ne_x2': np.float64(4.325748759210373), 'ne_half': np.float64(-2.1628743796051864)}. Best CPU head reconstruction: (np.float64(4.092082526742371), np.float64(0.31896668850946425), 'silu', 0.1, 'seconds', 'prefix_flags', 'silu'). Targets still depend on linear response/tail assumptions. No edits.

## Run 24
Testing all query-type indices and optional last-status embeddings. Best head lookup specification: (np.float64(4.092082526742371), np.float64(0.31896668850946425), 'silu', 'seconds', 'silu', 1, -1). Record targets differ under row versus stay weighting; official forward summary inspected. No edits; CPU reconstruction must be validated before deriving head edits.

## Run 25
Official readout is reached-row mean, not mean of stay means. Provisional record targets: {'insulin_x2': -12.769313194945848, 'insulin_half': 6.384656597472924, 'ne_x2': 4.504290667259787, 'ne_half': -2.2521453336298936}. Tested latent MC noise versus GPU mismatch: (np.float64(4.142977330371159), np.float64(0.8130057668094286), np.float64(0.32196663208413123), 'silu', 'seconds', 'silu', 1, -1). Best register permutation: (np.float64(2.406081454849649), 'seconds', 6, 2, (0, 3)). No edits.

## Run 26 / restart recovery
No edits. CPU-head disagreement exceeds estimated latent MC noise, so an edit based on that reconstruction is not yet justified. Recovering exact architecture, retained objects, state schema, and original rollout results before choosing the correction pathway.

## Run 27
Architecture confirms a separate action encoder feeding shared latent/recurrent pathways and code-specific last-value registers in the bin head. Recovering compact head/state details after run 26 output truncation. No edits.

## Runs 28–29
Run 28 had a syntax error and executed nothing. Corrected diagnostics in run 29: state identity, official reached-row summaries, exact head tensor dimensions, and no-last-reading cases. No edits.

## Run 30
State export identity verified (RMSE 7.4e-8). Official original reached-row shifts: I x2 -3.680, half +2.018; NE x2 +0.465, half -0.871. Original factual NLL 1.63164. Separately testing gap/prior/head activations and absent-register constants on no-last cases; results retained in r30_results and r30_mc_results. No edits.

## Run 31
No-last head search favors GELU gap/prior and SiLU value head; first-reading MC RMSE 3.16 versus expected noise 2.71, not yet a definitive match. Testing independent fine-bin boundary conventions and eight-column last register mappings on later readings. Retained r31_bins/r31_obsbins, r31_configs, and r31_best. No edits.

## Run 32
Register search reduced deterministic RMSE to 1.765; retained binning is left-boundary, not the source of mismatch. Full-sample latent MC checks and event/presence error diagnostics retained in r32_results/r32_details. Best spec retained as r32_best. No edits.

## Run 33 / restart recovery
Best CPU decoder MC disagreement is approximately sampling noise; recovering exact architecture, retained functions and action/state tensors to design a factual-preserving causal correction. No edits yet.

## Run 34
Recovered architecture: actions enter posterior and prelude, while a shared nonlinear value head reads the state plus code/register. Inspecting exact editable matrices and existing state coverage before choosing a residual-action edit. No edits.

## Run 35
Run 34 output was truncated by function disassembly. Retrieving compact posterior/head tensor names, normalization, state schema and retained feature returns. No tensors edited; forward 3/50.

## Run 36 / consolidated recovery
No edits; forward 3/50, Python 36/300. Original factual value NLL 1.63164. Original reached-row rollout effects: insulin x2 -3.680, half +2.018; NE x2 +0.465, half -0.871. Record innovation/RD evidence supports immediate insulin about -1.95 per unit and NE about +3.1 per rate unit. Eight-lag kernels in kernels19; provisional hour-12 row-weighted targets -12.77,+6.38,+4.50,-2.25, with tail/linearity uncertainty. CPU decoder best r32_best has GELU gap/prior/head, seconds log1p /10, measurement type 1; last numeric has flags column0, age column3, value z column6, absent rate/status bin0. MC errors approximately expected sampling noise. Existing 100 tuning-stay states in /work/fwd_3/states_original.npz (x4356x992), matched 4751 factual readings in d22; states are [out192,h768,z32]. Considering low-rank residual-action correction in z_post or value decoder, not an unconditional dose rescaling. Exact posterior z_post.0 is 192x576 [ctx,e,u], z_post.2 is 64x192. No gradients or optimizer. Training state IDs in state_train_ids36.

## Runs 37–38
Run 37 stopped at a raw-event MultiIndex lookup before testing the encoder. Run 38 uses canonical integer time keys and retries. Candidate results in action_candidates37. No edits; forward 4/50.

## Run 39
No edits. Run 38 posterior reconstruction errors remain material (insulin 0.076, NE 0.119). Recovering exact encoding, state timing, retained functions, and forward-4 schema before calculating an edit.

## Run 40
Architecture explicitly includes END atoms for infusion records. Checking whether omitted infusion ends explain the action encoder mismatch; also inspecting vocabulary conventions and exact context computation. No edits.

## Run 41
Run 40 output was truncated before record/count diagnostics. Repeating only compact bundle accounting and embedding shapes. No edits.

## Run 42
Exported counts exactly include WORLD starts plus infusion END atoms. Searching action encoder conventions on no-END bundles first. Best configurations: [(0.08681594580411911, 'gelu', 'gelu', (3, 1), (4, 3)), (0.0874592736363411, 'gelu', 'gelu', (3, 0), (4, 3)), (0.08748871833086014, 'gelu', 'gelu', (0, 2), (4, 3))]. No edits.

## Run 43
No-END reconstruction still mismatched (best RMS 0.087). Testing numeric layouts, kq embedding, and empty measurement encoding. Best: [(0.04662923514842987, ('interleaved_zfirst', ('value', 'dose', 'rate')), 0, 0, 'encoded_empty'), (0.04662923514842987, ('interleaved_zfirst', ('value', 'rate', 'dose')), 0, 0, 'encoded_empty'), (0.04818742722272873, ('interleaved_zfirst', ('value', 'dose', 'rate')), 0, 1, 'encoded_empty')]. No edits.

## Run 44
Independent activation/context search after no-END RMS 0.0466. Best: (0.04658462479710579, 'gelu', 'gelu', 0, ('gelu', 'log1p'), 'silu', 'gelu'). No edits.

## Run 45
Float32-boundary/unit search best: (2.107061845890712e-05, ('float32', 0, 'empty', (3, 2), (4, 3))). No edits.

## Run 46 / restart recovery
Run 45 reproduced no-END action posterior means with RMS 2.1e-5. Critical conventions: float32 bin boundaries, kq 0, empty measurement encoding, insulin units (3,2), NE units (4,3). Recovering validated functions and full update equations before a residual-dose edit. No tensors edited; forward passes 4/50.

## Run 47
Recovered full architecture. Actions affect both latent posterior and prelude attention; an unconditional embedding rescale is not justified. Recovering encoder dependencies and early causal identification notes to construct a residual-dose correction. No edits.

## Run 48
Threshold and innovation evidence, not rejected simple SSM fits, remain the basis for causal targets. Recovering exact action metadata and all-bundle export schema to validate END atoms before fitting any factual-preserving edit. No edits.

## Run 49 / all-action reconstruction
Generic start encoder test: (1.0064547240062893e-07, 2). Best END convention: (0.0728285163640976, ('zero', 'zero', 2, 0)). Validated bundle counts for 1602 drug bundles. No tensors edited.

## Run 50 / END partition test
Generic WORLD start encoder matches validated encoder to 1e-7; start event-kind index is 2. Treating all END atoms as actions failed (best RMS 0.0728285). END-as-observation best: (2.104543636960443e-05, ('missing', 'zero', 2, 0)). No weight edit until all-bundle reconstruction is validated.

## Runs 53–56 / recovery and measurement bin audit
Recovered architecture and causal evidence. No tensors edited. Testing float32 measurement boundaries, independently by bundle type. Best posterior test: (0.051534343510866165, False, 'left', 1); errors retained in err56.

## Run 57 / empty-action audit
Float32 measurement boundaries did not change errors. All 1602 action bundles match to ~2.1e-5 RMS; errors are on no-action bundles. Empty-action candidates: [('current', 0.051534347236156464), ('zero', 0.051534347236156464), ('encoded_empty', 0.012977438047528267)]. No tensors edited.

## Run 58 / corrected measurement search
The GPU uses encoded-empty u, contrary to the architecture's 'zero if none' statement. Correcting that reduces posterior RMS from .05153 to .01298. Measurement search best=(0.012922094203531742, 3, 2, 1, 'left'). No tensors edited.

## Run 59 / targeted remaining-encoder audit
Measurement categorical candidates: [('base', 0, 0.012977438047528267), ('embed.categorical.discharged_to.weight', 1, 0.013058122247457504), ('embed.categorical.action.weight', 1, 0.013269678689539433), ('embed.categorical.route.weight', 1, 0.013551345095038414)]; per-code unit fit {'glucose_finger': 0, 'glucose_lab': 5, 'map_arterial': 2, 'map_cuff': 2}. Action-only local posterior already validated; no tensors edited.

## Run 60 / encoders validated
Per-code measurement units are finger=0, lab=5, arterial=cuff=2. All-bundle posterior RMS=0.00003553; e_valid60/u_valid60 retained. Next: validate recurrent transition and construct an analytic, policy-residual correction. No weights edited.

## Run 61 / restart recovery
Posterior reconstruction is validated to 3.55e-5 RMS. Recovering recurrence equations and original causal evidence before choosing a factual-preserving residual-dose weight edit. No tensors edited.

## Run 62
Recovered architecture and reviewed rejected versus accepted causal estimates. Retrieving compact state/function schema after output truncation. No edits.

## Run 63 / recurrence reconstruction
Implemented no-gradient one-step recurrence and compared memory-age, activation and current-memory conventions against exported states. Best candidate: (0.0016396890860050917, 1.0700925486162305e-05, 'gelu', True, 'seconds', 1.0, 'both'). No edits.

## Run 64
Best recurrence uses GELU, includes the current e in memory, and adds core.age(log1p(age_seconds)) to both cross-attention keys and values. Out-state reconstruction RMS is approximately 1e-5. Inspecting interfaces for an analytic head-only correction before altering recurrence. No edits.

## Run 65 / decoder and action interfaces
Verifying generic dose re-encoding and exact value-register constants. Preparing a least-squares head-only residual-action correction, with factual features as a preservation constraint. No tensors edited.

## Runs 66–67
Run 66 stopped before editing because rate/dose unit table indices were reversed. Corrected dose-unit table I=3/NE=4, rate-unit table I=2/NE=3; verifying generic encoder, decoder and prediction-row alignment. No edits.

## Run 68 / recovery
Run 67's generic register is exact, but its explicit action encoder check has RMS 1.0097 and cannot support an edit. Recovering the previously validated encoder, recurrence, decoder, and record-based causal estimates. No edits; forward passes 4/50.

## Run 69
Architecture recovered; run 68 output truncated before function inventory. Retrieving compact retained interfaces and accepted causal kernels. No edits.

## Runs 70–71 / pooling audit
Explicit and trusted generic atom embeddings agree to 3.1e-8, but pooled u differs from u_valid60 by RMS 1.0097. Run 70 stopped at a metadata inspection error. Localizing pooling versus bundle alignment; no edits.

## Run 72 / action-owner alignment fix
Located run 65–67 mismatch: owners65 addresses the 1602 drug-bundle subset, not the 4491 full sequence. Remapped through drug_indices49 and rebuilt full counts. Testing against validated u_valid60. No edits.

## Runs 73–74
Run 73 stopped because transition64 requires age shape [N,128,1]. Corrected scalar axis and repeated state validation and factual-feature construction. No weights edited; forward passes 4/50.

## Run 75 / recurrent memory-mask audit
Posterior RMS=2.15e-5; factual decoder means track the observed means. Recurrent out RMS=.1306 and h RMS=12.29 are not acceptable. Testing mask polarity and inspecting retained validated memory inputs before using counterfactual state features. No edits.

## Run 76 / mask resolved
True denotes valid memory entries. With that convention, one-step out RMS is 1.51e-5 and h RMS is .00251. Recovering causal targets and validated interfaces before computing a factual-preserving dose correction. No edits.

## Runs 77–80 / head-only plan
Recovered accepted DML/RD insulin anchor ~-1.95, NE ~+3.1 and cross-sensor dynamic kernels. Provisional stay-weighted x2 targets I=-10.675, NE=+4.326; row-weighted I=-12.769, NE=+4.504. Simple SSM targets remain rejected. Plan: analytically fit a final value-head correction using dose-altered hidden features and factual preservation constraints. No edits yet.

## Run 84 / restart recovery
Run 83's head-only ridge edit cannot reach the causal slopes without factual loss: lambda=1 held-out dNLL=.001114, slopes I~-0.89/NE~.43-.58; lambda=.3 held-out dNLL=.006454 exceeds the .005 limit. Recovering fit interfaces before choosing a more selective edit. No tensors edited.

## Run 85 / inspect head objective and causal evidence
Inspecting the existing fit, original dose-rollout output, and latent-update interfaces to distinguish a weak head-only parameterization from a target/alignment error. No edits.

## Run 88 / latent-direction finite-difference probe
Original tuning rollout summaries: I2=-3.0206, Ihalf=1.6282 (60 reached stays); NE2=.4330, NEhalf=-.8431 (26 reached stays). Record-kernel targets remain provisional I2~-10.675/NE2~4.326. Probing sensor-selective latent directions with no gradients and no edits.

## Run 89 / reject unvalidated latent probe
Run 88 baseline RMSE=25.09, so its near-zero sensitivities and large latent direction are not evidence. Checking decoder return shapes and state alignment; no edits.

## Run 90 / recurrence recovery
Run 89 posterior RMS=1.95e-5 but recurrent out RMS=.1299 and h RMS=11.94. Latent probe remains invalid; recovering architecture and validated memory inputs before any edit. No tensors edited.

## Run 91 / targeted memory audit
Architecture reviewed. Posterior matches, so checking transition inputs and mask polarity rather than interpreting the invalid latent probe. No tensors edited.

## Run 92 / mask-polarity validation
State and prediction errors by mask: {'as returned': (0.12986552715301514, 11.935111045837402, 25.08648681640625), 'reversed': (1.4949718206480611e-05, 0.0025424514897167683, 0.0007484268862754107)}. Checking whether existing counterfactual features already use the correct convention. No tensors edited.

## Runs 93–94 / corrected latent probe
Run 93 stopped on pandas indexing before probing. Run 94 baseline RMSE=0.000727549; unit-effect direction results={'insulin': {'norm': 0.0, 'linear_rmse': 0.7071067811865476, 'nonlinear_rmse': 0.7071067811865476}, 'norepinephrine': {'norm': 2021.6579205609878, 'linear_rmse': 0.7000000011941472, 'nonlinear_rmse': 0.7072699565624382}}. No tensors edited.

## Run 96 / value-head input reconstruction
Hidden reconstruction RMS=0. Generated validated one-step head inputs for both drugs at factors 0,.5,1.5,2,3. Original dose slopes: {('insulin', 0.0): -0.37897342443466187, ('insulin', 0.5): -0.3507140576839447, ('insulin', 1.5): -0.3287563920021057, ('insulin', 2.0): -0.2550761103630066, ('insulin', 3.0): -0.1641843467950821, ('norepinephrine', 0.0): 0.17117874324321747, ('norepinephrine', 0.5): 0.29257407784461975, ('norepinephrine', 1.5): -0.005682266317307949, ('norepinephrine', 2.0): -0.023386526852846146, ('norepinephrine', 3.0): -0.019367428496479988}. No tensors edited.

## Run 97 / raw-head-input scalar correction
Fitted analytic ordinal bin-tilt scores using dose contrasts at 0,.5,1.5,2,3, with factual-zero regularization and stay-held-out validation. Results: {0.001: {'train_dNLL': 1.185721693243569, 'heldout_dNLL': 1.7496438853273455, 'coef_norm': 207.88876103546804, 'slopes': {('insulin', 2.0): -0.9745049185444954, ('insulin', 0.5): -0.7526900369443214, ('norepinephrine', 2.0): -0.245588024263237, ('norepinephrine', 0.5): 0.4942557754525222, ('insulin', 3.0): -0.6482577742968713, ('norepinephrine', 3.0): -0.14383401027586598}}, 0.003: {'train_dNLL': 0.34100809513489083, 'heldout_dNLL': 0.5394425445366878, 'coef_norm': 182.48661426219246, 'slopes': {('insulin', 2.0): -0.9155723646826897, ('insulin', 0.5): -0.6918848918456114, ('norepinephrine', 2.0): -0.08673923871587892, ('norepinephrine', 0.5): 0.24009646969896087, ('insulin', 3.0): -0.6090599080775653, ('norepinephrine', 3.0): -0.03495657066807687}}, 0.01: {'train_dNLL': 0.07565970883581126, 'heldout_dNLL': 0.1330940717274733, 'coef_norm': 146.4751571659859, 'slopes': {('insulin', 2.0): -0.8235920668936374, ('insulin', 0.5): -0.6381399145080828, ('norepinephrine', 2.0): -0.1532689072895136, ('norepinephrine', 0.5): 0.1262774614577874, ('insulin', 3.0): -0.5479250612052751, ('norepinephrine', 3.0): -0.06455488744348806}}, 0.03: {'train_dNLL': 0.017870054287421867, 'heldout_dNLL': 0.034725120740966044, 'coef_norm': 107.29819067743038, 'slopes': {('insulin', 2.0): -0.7606009577918071, ('insulin', 0.5): -0.6023317196295277, ('norepinephrine', 2.0): -0.16425190058408862, ('norepinephrine', 0.5): 0.047061491958779296, ('insulin', 3.0): -0.5050410425807533, ('norepinephrine', 3.0): -0.06972684894121806}}, 0.1: {'train_dNLL': 0.0027638256217535446, 'heldout_dNLL': 0.006999956049771413, 'coef_norm': 63.18266087819877, 'slopes': {('insulin', 2.0): -0.7088287095331086, ('insulin', 0.5): -0.5892877779434538, ('norepinephrine', 2.0): -0.18321425135777178, ('norepinephrine', 0.5): 0.003034811508722028, ('insulin', 3.0): -0.4699774096605705, ('norepinephrine', 3.0): -0.08291323591414959}}, 0.3: {'train_dNLL': -0.000254933712967499, 'heldout_dNLL': 0.0008573269092113806, 'coef_norm': 33.63802021035437, 'slopes': {('insulin', 2.0): -0.6741209945962788, ('insulin', 0.5): -0.5873503253195552, ('norepinephrine', 2.0): -0.2111859186193917, ('norepinephrine', 0.5): -0.007753561957243456, ('insulin', 3.0): -0.4489769492286717, ('norepinephrine', 3.0): -0.10595532806682603}}, 1.0: {'train_dNLL': -0.0006430335353646772, 'heldout_dNLL': -0.00036154402892997196, 'coef_norm': 17.24273558983572, 'slopes': {('insulin', 2.0): -0.6239073013746047, ('insulin', 0.5): -0.559181423813757, ('norepinephrine', 2.0): -0.24951436165373195, ('norepinephrine', 0.5): 0.0032665674889910687, ('insulin', 3.0): -0.4210671063130235, ('norepinephrine', 3.0): -0.13945267460081018}}, 3.0: {'train_dNLL': -0.000455294825703859, 'heldout_dNLL': -0.00034772700494614667, 'coef_norm': 9.534411117915507, 'slopes': {('insulin', 2.0): -0.5482489721083419, ('insulin', 0.5): -0.4943475801532573, ('norepinephrine', 2.0): -0.2528174604540175, ('norepinephrine', 0.5): 0.07253744835856203, ('insulin', 3.0): -0.37517776020752375, ('norepinephrine', 3.0): -0.14853855901368052}}}. No tensors edited.

## Run 98 / upstream repair recovery
Run 97 confirms that a factual-preserving raw-head correction cannot recover the supported drug slopes. Recovering architecture, causal evidence, and retained interfaces before testing an upstream dose-residual correction. No weights edited.

## Run 99 / compact interface recovery
Architecture confirms doses enter through the action encoder and posterior/prelude. Recovered notes support corrected DML effects I≈-1.94 and NE≈+3.05; rejected SSMs must not determine targets. Retrieving validated functions and edit locations. No edits.

## Run 100 / causal timing and selective-path audit
Inspecting retained causal response kernels and validated one-step dependencies. Next candidate must encode dose deviations from observational policy, rather than apply a broad value-head shift. No weights edited.

## Run 101 / counterfactual factor-one falsification
Factor-one checks: {'insulin': {'readings': 2340, 'logit_RMS': 0.00022519048070535064, 'mean_RMS': 0.000640077400021255, 'mean_max_abs': 0.003021240234375}, 'norepinephrine': {'readings': 1643, 'logit_RMS': 0.0002523489820305258, 'mean_RMS': 0.0008458229713141918, 'mean_max_abs': 0.00750732421875}}. Require factual equivalence before interpreting retained CF sensitivities or fitting edits. No weights edited.

## Runs 102–103 / boundary-safe prelude probe
Run 102 stopped at boundary assertion; excluded 100 bundles. Prelude reconstruction RMS=5.348165643681568e-08. Extractor diagnostics={'insulin': {0.001: {'contrast_RMSE': 1.074684913590839, 'target_RMS': 1.8024748232449397, 'factual_signal_SD': 7.060348895313207, 'coefficient_norm': 239.78112500593707}, 0.01: {'contrast_RMSE': 1.1360828124962832, 'target_RMS': 1.8024748232449397, 'factual_signal_SD': 2.5384522372148277, 'coefficient_norm': 225.90311654916917}, 0.1: {'contrast_RMSE': 1.2045028662832789, 'target_RMS': 1.8024748232449397, 'factual_signal_SD': 0.7661904940667225, 'coefficient_norm': 176.408822009451}, 1.0: {'contrast_RMSE': 1.2588084156226897, 'target_RMS': 1.8024748232449397, 'factual_signal_SD': 0.38577482731695006, 'coefficient_norm': 107.58646671893251}}, 'norepinephrine': {0.001: {'contrast_RMSE': 0.3595802474150394, 'target_RMS': 0.5438149097874485, 'factual_signal_SD': 2.821748022507915, 'coefficient_norm': 117.07558839029842}, 0.01: {'contrast_RMSE': 0.3877690805266286, 'target_RMS': 0.5438149097874485, 'factual_signal_SD': 0.8650247893908315, 'coefficient_norm': 107.58006161694054}, 0.1: {'contrast_RMSE': 0.4130780770235559, 'target_RMS': 0.5438149097874485, 'factual_signal_SD': 0.2973784476320589, 'coefficient_norm': 86.5835975851364}, 1.0: {'contrast_RMSE': 0.44348336828539225, 'target_RMS': 0.5438149097874485, 'factual_signal_SD': 0.13367383012881703, 'coefficient_norm': 45.30809326099209}}}. No weights edited.

## Run 104 / posterior-input residual feasibility
Compared raw and hidden posterior features: {('raw', 'insulin', 0.01): {'contrast_RMSE': 0.3664, 'factual_SD': 0.3935, 'dose_slopes': {0.0: 0.981, 0.5: 0.996, 1.5: 0.979, 2.0: 0.982, 3.0: 0.894}}, ('raw', 'insulin', 0.1): {'contrast_RMSE': 0.3692, 'factual_SD': 0.3375, 'dose_slopes': {0.0: 0.979, 0.5: 0.984, 1.5: 0.974, 2.0: 0.98, 3.0: 0.89}}, ('raw', 'insulin', 1.0): {'contrast_RMSE': 0.3888, 'factual_SD': 0.2819, 'dose_slopes': {0.0: 0.955, 0.5: 0.902, 1.5: 0.931, 2.0: 0.954, 3.0: 0.86}}, ('raw', 'norepinephrine', 0.01): {'contrast_RMSE': 0.0989, 'factual_SD': 0.111, 'dose_slopes': {0.0: 0.92, 0.5: 0.933, 1.5: 1.0, 2.0: 1.027, 3.0: 0.922}}, ('raw', 'norepinephrine', 0.1): {'contrast_RMSE': 0.0992, 'factual_SD': 0.1027, 'dose_slopes': {0.0: 0.91, 0.5: 0.906, 1.5: 1.003, 2.0: 1.025, 3.0: 0.92}}, ('raw', 'norepinephrine', 1.0): {'contrast_RMSE': 0.109, 'factual_SD': 0.0775, 'dose_slopes': {0.0: 0.84, 0.5: 0.719, 1.5: 1.026, 2.0: 1.015, 3.0: 0.904}}, ('hidden', 'insulin', 0.01): {'contrast_RMSE': 0.9997, 'factual_SD': 1.2385, 'dose_slopes': {0.0: 0.906, 0.5: 0.811, 1.5: 0.895, 2.0: 0.738, 3.0: 0.492}}, ('hidden', 'insulin', 0.1): {'contrast_RMSE': 1.0173, 'factual_SD': 0.6816, 'dose_slopes': {0.0: 0.903, 0.5: 0.79, 1.5: 0.865, 2.0: 0.702, 3.0: 0.467}}, ('hidden', 'insulin', 1.0): {'contrast_RMSE': 1.067, 'factual_SD': 0.3996, 'dose_slopes': {0.0: 0.869, 0.5: 0.656, 1.5: 0.737, 2.0: 0.586, 3.0: 0.393}}, ('hidden', 'norepinephrine', 0.01): {'contrast_RMSE': 0.3374, 'factual_SD': 0.5299, 'dose_slopes': {0.0: 0.497, 0.5: 0.704, 1.5: 1.003, 2.0: 0.705, 3.0: 0.427}}, ('hidden', 'norepinephrine', 0.1): {'contrast_RMSE': 0.3494, 'factual_SD': 0.2632, 'dose_slopes': {0.0: 0.487, 0.5: 0.712, 1.5: 0.904, 2.0: 0.634, 3.0: 0.384}}, ('hidden', 'norepinephrine', 1.0): {'contrast_RMSE': 0.3724, 'factual_SD': 0.1377, 'dose_slopes': {0.0: 0.39, 0.5: 0.595, 1.5: 0.661, 2.0: 0.471, 3.0: 0.288}}}. Prelude-only extraction is too lossy; earlier features may retain treatment/history separation. No weights edited.

## Run 105 / restart recovery
Raw posterior-input residuals preserve dose contrasts much better than hidden posterior features. Recovering architecture, causal evidence, and interfaces before constructing a selective upstream edit. No weights edited.

## Runs 106–109 / causal and latent-path recovery
Accepted immediate effects: I≈−1.94; NE≈+3.05. Record impulse kernels support delayed/declining insulin and faster-decaying NE responses. Existing z-injection finite differences are nearly insensitive, so a posterior-only edit is insufficient. Checking unused latent channels and drug-bundle gating before constructing a new selective state route. No edits.

## Run 110 / direct latent-token controllability
Testing 48 orthonormal token directions with central finite differences (±2), using validated recurrence and decoder. No weights saved. This separates a weak existing z-coordinate route from weak token controllability.

## Run 112 / selective-route recovery
Large latent perturbations change slots but barely change predictive output; recovering the architecture and validated helpers before choosing a dose-residual route. No weights edited.

## Run 113 / latent-route feasibility
Recovered architecture: dose residuals can enter z_post, but must propagate through z_in/prelude and recurrence. Inspecting token controllability and validated causal/decoder interfaces before an analytical edit. No edits.

## Run 115 / attention and state diagnostics
Slots have RMS≈49 versus latent-token RMS≈0.24. Measuring head contributions, attention masses, and low-variance state directions to locate a minimally disruptive dose-residual channel. No weights edited.

## Run 116 / attention dose-residual extraction
Action attention is strong (head 2≈0.78, head 3≈0.66); latent attention is negligible. Tested analytical dose-contrast extractors on the four attention heads, penalizing factual signals and including cross-drug zero targets. Results: {(0, 0.03): {'factual_RMS': [1.363, 0.354], 'contrast_RMSE': [0.562, 0.131], 'slopes': {('insulin', 0.0): 0.913, ('insulin', 0.5): 0.945, ('insulin', 2.0): 0.921, ('insulin', 3.0): 0.805, ('norepinephrine', 0.0): 0.864, ('norepinephrine', 0.5): 0.878, ('norepinephrine', 2.0): 1.041, ('norepinephrine', 3.0): 0.871}}, (0, 0.3): {'factual_RMS': [0.736, 0.196], 'contrast_RMSE': [0.675, 0.166], 'slopes': {('insulin', 0.0): 0.873, ('insulin', 0.5): 0.868, ('insulin', 2.0): 0.841, ('insulin', 3.0): 0.705, ('norepinephrine', 0.0): 0.746, ('norepinephrine', 0.5): 0.723, ('norepinephrine', 2.0): 1.041, ('norepinephrine', 3.0): 0.797}}, (0, 3.0): {'factual_RMS': [0.347, 0.083], 'contrast_RMSE': [0.936, 0.24], 'slopes': {('insulin', 0.0): 0.754, ('insulin', 0.5): 0.533, ('insulin', 2.0): 0.588, ('insulin', 3.0): 0.47, ('norepinephrine', 0.0): 0.42, ('norepinephrine', 0.5): 0.328, ('norepinephrine', 2.0): 0.982, ('norepinephrine', 3.0): 0.661}}, (1, 0.03): {'factual_RMS': [1.485, 0.42], 'contrast_RMSE': [0.62, 0.145], 'slopes': {('insulin', 0.0): 0.914, ('insulin', 0.5): 0.937, ('insulin', 2.0): 0.897, ('insulin', 3.0): 0.768, ('norepinephrine', 0.0): 0.843, ('norepinephrine', 0.5): 0.851, ('norepinephrine', 2.0): 1.038, ('norepinephrine', 3.0): 0.846}}, (1, 0.3): {'factual_RMS': [0.79, 0.209], 'contrast_RMSE': [0.748, 0.189], 'slopes': {('insulin', 0.0): 0.874, ('insulin', 0.5): 0.815, ('insulin', 2.0): 0.799, ('insulin', 3.0): 0.648, ('norepinephrine', 0.0): 0.696, ('norepinephrine', 0.5): 0.689, ('norepinephrine', 2.0): 1.035, ('norepinephrine', 3.0): 0.751}}, (1, 3.0): {'factual_RMS': [0.335, 0.084], 'contrast_RMSE': [1.015, 0.261], 'slopes': {('insulin', 0.0): 0.752, ('insulin', 0.5): 0.482, ('insulin', 2.0): 0.537, ('insulin', 3.0): 0.41, ('norepinephrine', 0.0): 0.371, ('norepinephrine', 0.5): 0.314, ('norepinephrine', 2.0): 0.936, ('norepinephrine', 3.0): 0.614}}, (2, 0.03): {'factual_RMS': [1.159, 0.314], 'contrast_RMSE': [0.453, 0.119], 'slopes': {('insulin', 0.0): 0.968, ('insulin', 0.5): 0.988, ('insulin', 2.0): 0.963, ('insulin', 3.0): 0.846, ('norepinephrine', 0.0): 0.892, ('norepinephrine', 0.5): 0.884, ('norepinephrine', 2.0): 1.038, ('norepinephrine', 3.0): 0.887}}, (2, 0.3): {'factual_RMS': [0.688, 0.18], 'contrast_RMSE': [0.553, 0.149], 'slopes': {('insulin', 0.0): 0.932, ('insulin', 0.5): 0.91, ('insulin', 2.0): 0.904, ('insulin', 3.0): 0.768, ('norepinephrine', 0.0): 0.789, ('norepinephrine', 0.5): 0.751, ('norepinephrine', 2.0): 1.04, ('norepinephrine', 3.0): 0.83}}, (2, 3.0): {'factual_RMS': [0.349, 0.076], 'contrast_RMSE': [0.824, 0.221], 'slopes': {('insulin', 0.0): 0.816, ('insulin', 0.5): 0.568, ('insulin', 2.0): 0.661, ('insulin', 3.0): 0.546, ('norepinephrine', 0.0): 0.5, ('norepinephrine', 0.5): 0.341, ('norepinephrine', 2.0): 1.017, ('norepinephrine', 3.0): 0.711}}, (3, 0.03): {'factual_RMS': [0.0, 0.002], 'contrast_RMSE': [1.644, 0.496], 'slopes': {('insulin', 0.0): 0.0, ('insulin', 0.5): 0.0, ('insulin', 2.0): 0.0, ('insulin', 3.0): 0.0, ('norepinephrine', 0.0): -0.0, ('norepinephrine', 0.5): -0.0, ('norepinephrine', 2.0): 0.0, ('norepinephrine', 3.0): 0.0}}, (3, 0.3): {'factual_RMS': [0.0, 0.001], 'contrast_RMSE': [1.644, 0.496], 'slopes': {('insulin', 0.0): 0.0, ('insulin', 0.5): 0.0, ('insulin', 2.0): 0.0, ('insulin', 3.0): 0.0, ('norepinephrine', 0.0): -0.0, ('norepinephrine', 0.5): -0.0, ('norepinephrine', 2.0): 0.0, ('norepinephrine', 3.0): 0.0}}, (3, 3.0): {'factual_RMS': [0.0, 0.0], 'contrast_RMSE': [1.644, 0.496], 'slopes': {('insulin', 0.0): 0.0, ('insulin', 0.5): 0.0, ('insulin', 2.0): 0.0, ('insulin', 3.0): 0.0, ('norepinephrine', 0.0): -0.0, ('norepinephrine', 0.5): -0.0, ('norepinephrine', 2.0): 0.0, ('norepinephrine', 3.0): 0.0}}}
No weights edited.

## Run 117 / direct-attention controllability
Head 2 preserves dose contrasts (I x2≈0.90, NE x2≈1.04 at penalty .3), but factual residual RMS≈.69/.18. Tested direct attention-output injection to distinguish weak latent attention from downstream suppression. Gradient RMS=0.0006628929630836985; fitted route norms=[15674.819786445529, 34275.906936299514]. No weights edited.

## Run 118 / private-state readout feasibility
Direct attention routes require norms≈15,675/34,276 and are rejected. Captured original pre-coda states and tested four low-variance state channels with covariance-penalized value-input readouts. State baseline RMS=[0.013631374845695361, 0.013241554470490004, 0.01301869408009892, 0.01310210398858578]; head baseline SD=[0.04169307177247208, 0.036776184078815034, 0.03405533666690212, 0.030618347900509577]. No weights edited.

## Run 119 / restart recovery
Private state directions have factual RMS≈0.013 and normalized readout SD≈0.03–0.04, with near-unit response across all four measurement types. Recovering architecture, causal evidence, and helpers to implement and validate a selective dose-effect state channel. No weights edited.

## Run 123 / pre-coda dose extraction
Private directions transmit through coda with gains≈0.998 and orthogonal RMS leakage≈0.001–0.003 per unit. Fit covariance-penalized linear dose extractors from factual and counterfactual pre-coda states, including x0/x0.5/x1.5/x2/x3. {0.01: {'factual_RMS': [2.7793, 0.9584], 'contrast_RMSE': [1.1027, 0.3927], 'coefficient_norms': [81.85, 28.7], 'slopes': {('insulin', 0.0): 0.905, ('insulin', 0.5): 0.803, ('insulin', 1.5): 0.701, ('insulin', 2.0): 0.564, ('insulin', 3.0): 0.386, ('norepinephrine', 0.0): 0.423, ('norepinephrine', 0.5): 0.688, ('norepinephrine', 1.5): 0.589, ('norepinephrine', 2.0): 0.405, ('norepinephrine', 3.0): 0.259}}, 0.1: {'factual_RMS': [0.8157, 0.2535], 'contrast_RMSE': [1.1803, 0.4186], 'coefficient_norms': [23.96, 6.91], 'slopes': {('insulin', 0.0): 0.827, ('insulin', 0.5): 0.687, ('insulin', 1.5): 0.625, ('insulin', 2.0): 0.5, ('insulin', 3.0): 0.334, ('norepinephrine', 0.0): 0.331, ('norepinephrine', 0.5): 0.57, ('norepinephrine', 1.5): 0.45, ('norepinephrine', 2.0): 0.313, ('norepinephrine', 3.0): 0.2}}, 1.0: {'factual_RMS': [0.3921, 0.1054], 'contrast_RMSE': [1.2321, 0.4342], 'coefficient_norms': [6.36, 2.13], 'slopes': {('insulin', 0.0): 0.662, ('insulin', 0.5): 0.573, ('insulin', 1.5): 0.524, ('insulin', 2.0): 0.422, ('insulin', 3.0): 0.28, ('norepinephrine', 0.0): 0.236, ('norepinephrine', 0.5): 0.42, ('norepinephrine', 1.5): 0.32, ('norepinephrine', 2.0): 0.226, ('norepinephrine', 3.0): 0.145}}, 10.0: {'factual_RMS': [0.1669, 0.0335], 'contrast_RMSE': [1.4011, 0.4669], 'coefficient_norms': [2.03, 0.56], 'slopes': {('insulin', 0.0): 0.29, ('insulin', 0.5): 0.262, ('insulin', 1.5): 0.231, ('insulin', 2.0): 0.19, ('insulin', 3.0): 0.127, ('norepinephrine', 0.0): 0.096, ('norepinephrine', 0.5): 0.166, ('norepinephrine', 1.5): 0.11, ('norepinephrine', 2.0): 0.078, ('norepinephrine', 3.0): 0.051}}}
No weights edited.

## Run 124 / reject passive pre-coda extraction
Passive pre-coda extractors attenuate x2/x3 contrasts too much and are rejected. Investigating a new low-rank wire from dose-preserving attention features through quiet prelude/state directions, with measured private-state persistence. No edits.

## Run 125 / private controller and attention wire
Existing private-state transition gains are near zero. Fitted a covariance-penalized controller from finite state perturbations and an attention-output reconstruction after replacing two head-2 value channels. Variables P125/T125, QE125/SP125, Vnew125/Wrepair125/chosen125 hold the proposed analytical components. No weights saved.

## Run 130 / compact recovery
Recovered accepted immediate effects I≈-1.95 and NE≈+3.1, insulin supported by independent RD; rejected SSM targets are not used. Eight-lag kernels agree approximately across sensors and reject simple AR1. Provisional hour-12 targets remain -12.77,+6.38,+4.50,-2.25. Inspecting exact interfaces to assemble a reversible four-state correction. No edits saved; forward passes 4/50.

## Reconstruction-only attention edit staged
Saved original keys plus Vnew125 and Wrepair125. No causal injection or value-head change. Next: paired factual audit; undo if reconstruction harms fit. CPU cache remains original.

## Run 133 / reconstruction factual audit
Forward 5 (files fwd_6): 100 tuning stays, 4,751 values. Original and reconstruction-only edited NLL both 1.63164; all reported event NLL/MAE match. Reconstruction accepted provisionally. Recovering fitted controller/wire interfaces before causal injection. Forward passes 5/50; only attention kv/out tensors edited.

## Runs 139–145 / recovered controller identities
Recovered architecture and causal kernels; rejected SSM remains excluded. P125 decodes flowed private-state perturbations with RMS≈0.012; original private persistence is near zero. T125 satisfies P125.T@T125=I and is a write right-inverse, not an original-core inverse. SP125 equals QE125/core.prelude_norm.weight. Attention dose channels have factual SD≈0.688/0.180. Restored controller is Ccontroller145; clean and regularized write routes are routes145. No additional weights saved; reconstruction-only attention edit remains factual-NLL neutral.

## Runs 146–152 / recovered repair geometry
Ccontroller145 = Cperp145 + R118@P125.T; Cperp is the original c_out outside the four private directions. Uexact145 satisfies Cperp@U=0 and P.T@U=I, allowing an exact private write through b_in. No new weights saved. Fitting two-state record-kernel approximations and calibrating the attention wire locally.

## Run 159 / factual-anchored controller
Recovery found controllers153 used centered factual covariance but had large unpenalized means (about 30 in one channel). They are not safe to submit. Refit private c_out correction relative to the original mapping using complete raw factual slot states. controllers159 and routes159 contain the new candidates. No additional weights saved.

## Run 160 / controller and dose wire staged
Used factual-anchored controllers159[.01]; private states receive one raw-rate unit in both poles for each drug. Saved c_out, b_in, and wire_out152, retaining the reconstruction kv edit. Value head is untouched. Pending paired factual audit; these are not yet validated causal predictions.

## Run 162 / staged controller rejected
Forward 6 (fwd_7), 100 tuning stays: original value NLL 1.63164; controller+wire edited NLL 1.76833, delta +0.13669. Rejected before causal readout. Dropped b_in/c_out edits and restored the previously NLL-neutral reconstruction-only attention out Wrepair125; reconstruction kv retained. Need diagnose recurrent/off-manifold leakage.

## Run 165 / controller isolation
Staged c_out=C160 alone alongside neutral attention reconstruction; b_in original, no dose wire or readout. This isolates recurrent-controller drift.

## Run 166 / controller-only rejected
Forward 7 (fwd_8): c_out controller alone gives value NLL 1.71495 vs 1.63164 (+.08331). Dropped c_out. Large factual degradation is not solely dose-wire magnitude. Investigating omitted context/posterior feedback and multi-bundle controller stability; only neutral reconstruction remains.

## Run 167 / one-step quiet write, no persistence controller
Abandoned harmful c_out controller. Original c_out retained; analytical right-inverse U=solve(C_original,R118) writes a 0.1-scale dose feature after recurrent blocks via b_in. Staged B167 and wire_out152 with reconstruction kv. Value head untouched. Pending factual isolation audit.

## Run 168 / restart recovery
One-step b_in write remains unvalidated. Recovering architecture, causal evidence, and the existing 100-stay audit set before the paired factual test. No additional weights changed.

## Forward 8 / quiet write factual audit
100 tuning stays, 4751 readings: original NLL 1.63164, edited 1.63170 (delta about +0.00006). Quiet b_in write accepted provisionally. No causal value readout exists yet; next recover decoder features and fit a factual-anchored dose correction.

## Run 173 / edited training-state decoder recovery
Forward 9 supplied 4491 edited states on the original 100-stay training feature sample. Aligned by stay/hour; CPU cache now uses the three current edits. Recovering factual decoder features and x1/x2 one-step contrasts. No new weights saved.

## Run 174 / decoder recovery
Quiet dose write remains factual-neutral on the prior audit (+~0.00006 NLL). Recovered architecture, empirical-effect evidence, and live helper interfaces before fitting a factual-anchored value readout. No new edits.

## Run 178 / generalized contrast features
Collected current quiet-write decoder features for both drugs at x0, x0.5, x1.5, x2, and x3. Retained record kernels19; no new weights edited. Next fit a factual-anchored, code-selective final value readout.

## Runs 179–180 / reject final-layer-only fit
Closed-form final value-layer candidates could not attain accepted effects within factual +0.005 NLL. Lambda 1 gave validation +0.00184 but slopes only I~−1.03/NE~+0.39 at x2; stronger edits violated fit. No weights saved. Captured all dose contrasts before the value GELU to diagnose the private wire.

## Run 185 / conservative value-readout staged
Recovered original rollout: I x2 -3.680, I half +2.018; NE x2 +0.465, NE half -0.871. Accepted records-based hour-12 targets remain approximately -12.77,+6.38,+4.50,-2.25; rejected early SSM targets are not used. Staged readout179[1.0] in value_bin.2 weight and bias on top of the three quiet-wire edits, pending paired factual GPU validation. No upstream controller or other head was changed.

## Forward 10 / full-strength readout fails factual limit
100 tuning stays, 4751 readings: original NLL 1.63164, edited 1.63710 (+0.00546), so readout179[1] is rejected at full strength. Run 186 stages 0.70 times that correction, retaining the prior factual-neutral quiet wire, pending a new paired audit.

## Run 187 / restarted recovery
Forward 11 (fwd_12): reduced readout at 0.70 strength gives factual NLL 1.63427 vs original 1.63164, delta +0.00263 on 100 tuning stays / 4751 readings. Within the limit on this sample; causal rollout benefit remains unvalidated. Recovering architecture, live interfaces and stay IDs before paired rollouts.

## Forward 12 / weak causal benefit
Paired rollout on the 100-stay audit: original shifts [-3.680, +2.018, +0.465, -0.871]; current shifts [-4.302, +2.434, +0.496, -0.912] for I x2/half, NE x2/half. Current factual delta +0.00263. Final-layer-only repair is insufficient, especially for NE. Investigating selective head-feature construction rather than increasing its scale.

## Run 192 / pre-GELU feature diagnosis
Fitting factual-anchored linear dose decoders across x0, half, x1.5, x2, x3 and a dose-invariant sensor-family gate. No new weights saved; current conservative edit retained.

## Runs 194–200 / nonlinear-head recovery
Recovered architecture and current quiet-wire feature caches. Sensor-family gate is nearly exact. Audited raw dose units: contrasts are correct physical rates; targets179 are logit corrections, not full logits (the apparent huge target means in run 193 were a misinterpretation). Rebuilt canonical dose directions directly from verified head inputs. No model weights changed during recovery.

## Runs 201–207 / verified raw-head geometry
Recovered exact raw head inputs (hidden reconstruction RMS 2.9e-7). Sensor-family gate is nearly exact (+1 glucose, -1 MAP). Canonical cases are paired intervened/factual raw inputs; their factual cache differs by only 3e-5 RMS from X200. Using paired differences and fitting family-specific dose directions. No weights changed; conservative five-tensor edit retained.

## Runs 208–212 / nonlinear repair setup
Recovered canonical paired raw head inputs, family masks and dose directions. No additional weights changed. Next fit analytically with a few replaced GELU units, preserving original factual logits while enforcing record-derived dose contrasts.

## Run 219 / nonlinear head candidate staged
Staged linear2 eta=1: replaced two first-layer GELU units and fitted the final value layer analytically. Offline train/validation NLL deltas +.00481/+.00178; x2 immediate slopes I -1.712, NE +1.465. All prior edits retained in backup219 for rollback. Pending paired GPU factual audit; no causal success claimed.

## Forward 13 / nonlinear eta=1 rejected
100-stay tuning GPU factual NLL 1.64576 vs 1.63164 (+.01412), despite offline validation +.00178. Rejected and rolled back to backup219 (five-tensor conservative edit, factual +.00263). Need diagnose dose-direction generalization / insufficient distillation coverage; no causal rollout tested for this candidate.

## Run 222 / independent-coverage diagnosis
Recovered architecture and fitting interfaces after rollback. Current five-tensor conservative edit remains staged; no new weights saved. Investigating whether nonlinear dose-feature generalization caused the offline/GPU factual discrepancy before attempting another candidate.

## Run 228 / independent factual reconstruction
Reconstructed 4751 tuning readings from original and quiet-wire states. Offline original NLL 1.623310; conservative delta 0.003125; rejected nonlinear delta 0.013378. No weights changed. Exact fine-bin boundaries use side='left'.

## Run 234 / broader factual anchoring
Refit two-unit and gated four-unit analytic value heads using the original training cache plus independent tuning factual states. No weights saved; comparing factual NLL and paired immediate-dose slopes.

## Run 235 / broader-anchored gated head staged
Staged pair4 lambda=.1, etaI=etaNE=.3. Replaced four first-layer GELU units and analytically fitted final value weights/bias; upstream quiet dose wire unchanged. Offline factual deltas 0.001688/0.003719/0.002557; immediate slopes [-1.5882333346430055, -1.3778477358669075, 1.970496598363532, 1.7545535450012104]. Backup235 preserves conservative edit. Pending paired GPU factual audit.

## Runs 237–238 / gated head validated, partial causal repair
pair4 lambda=.1 etaI=etaNE=.3: GPU factual NLL 1.63421 vs 1.63164, delta +.00257 on 100 tuning stays. Paired rollout shifts I x2 -7.878, I half +3.322, NE x2 +2.231, NE half -1.038 (original -3.680,+2.018,+.465,-.871). All four improve toward records-derived estimates but remain too small. backup237 preserves this seven-tensor validated candidate. Run237 stopped during backup before making changes; corrected using tensor.clone().

## Run 247 / restart tensor verification
Independent 100-stay factual audit: delta +.00393. Forward18 reproduces [-7.878,+3.322,+2.231,-1.038] exactly. Exploratory grid settings must not be confused with saved tensors; comparing actual final-head weights against T235 and grid242 before staging any new candidate.

## Run 248 / stronger head actually staged
Verified prior current head equals T235 (.3/.3); grid242 candidates were only exploratory. Saved grid242[(.5,.4)] final weights and bias; four-unit first layer and quiet upstream wire unchanged. backup248 preserves the validated .3/.3 candidate. Offline factual deltas +.002136/+.003901/+.003925; immediate I x2 slope -1.725, NE +1.961. Needs independent factual and paired rollout validation.

## Run 249 / stronger-head audit recovery
Forward19 factual: original 1.61582, edited 1.62034, delta +0.00452 on 4664 readings / 100 independent tuning stays. Within tolerance but close to limit. Current grid242[(.5,.4)] causal rollout remains unvalidated; recovering evidence and live interfaces before further changes. No weights changed.

## Forward20 / stronger-head second-sample rollout
On tuning IDs100–199: original [-4.072,+2.273,+.615,-.858], edited [-8.523,+4.138,+1.781,-1.103]. All four improve in the records-supported direction; effects remain too small, especially NE. Current factual delta +.00452 on these stays. Investigating factual-error allocation before strengthening. No weights changed.

## Runs 252–254 / conditional factual compensation
Constant sensor errors are small; testing regularized original-logit residuals conditioned on prior sensor bin, fitted on training fitmask only. Run253 stopped on structured vocabulary lookup before fitting; run254 corrects it. No weights changed.

## Run 254 / conditional compensation rejected
Prior-bin original-logit offset candidates did not improve independent tuning NLL (best ~+.00399 versus +.00393 current); none saved. Reproduced .5/.4 analytic final-head solution to 2.8e-12. Investigating coupling of the two dose constraints.

## Run 260 / dose-constraint coupling
Recovered the analytic fit. Existing grid confirms stronger insulin weighting reduces the NE response. Tested antisymmetric GELU-pair readouts and penalties on dose-contrast changes through older features, using closed-form regressions. No weights saved; current .5/.4 candidate remains staged.

## Runs 261–262 / broaden dose-decoder anchoring
Rejected antisymmetric-pair readout: independent offline factual delta >.010. Old insulin decoder factual SD is 1.12 on fitting data but 2.79 on independent states. Fit small linear decoders analytically with train-fit and independent factual covariance penalties plus physical paired-dose contrasts. No model weights changed.

## Run 264 / restart recovery
Run263 broadened-decoder .3/.3 has offline factual deltas +.001399/+.004756/+.004258 and stronger NE immediate slopes. Larger settings exceed the factual budget. No weights changed; recovering architecture and live fitting objects before validation.

## Run 267 / broadened-decoder candidate staged
Verified current head exactly matches grid242 .5/.4 and its old pair4 rows. Backed up all seven edits as backup267. Staged arch263 rows and trial263 .3/.3 final weights/bias; quiet upstream wire unchanged. Offline NLL deltas +.001399/+.004756/+.004258; immediate slopes [-1.809,-1.905,+3.126,+2.045]. Pending independent GPU factual audit.

## Forward21 / broadened .3/.3 rejected
Independent 100-stay GPU factual delta +.00535 (1.62117 vs 1.61582), above tolerance. Saved rejected tensors in backup268_rejected and restored backup267, the validated old .5/.4 edit. Deriving weaker broadened-decoder solutions analytically; no rollout claim for the rejected candidate.

## Run 269 / weaker broadened-head grid
Reproduced run263 solution to <1e-7: T=T0+solve(GF+etaI*GDI+etaNE*GDN+1e-4 I, RF+etaI*RDI+etaNE*RDN)/feature_scale. Computed weaker candidates and offline scores, without saving any. Validated old .5/.4 head remains current.

## Run 270 / weaker broadened head staged
Staged new four-unit decoder rows with grid269 (.25,.15) final head. Offline factual deltas +.000705/+.003921/+.003538; slopes [-1.826,-1.889,+2.559,+1.723]. Quiet wire unchanged. backup267 preserves validated old .5/.4; GPU audit pending.

## Run 273 / weaker broadened head validated
Current arch263 + grid269 (.25,.15), hash 58b516f6e0.
Forward22 (fwd_23), tuning stays100–199: factual delta +.00409
(edited1.61991 vs original1.61582), within tolerance.
Forward23 (fwd_24), same stays: shifts [-10.337,+5.389,+1.830,-.880],
original [-4.072,+2.273,+.615,-.858].
Insulin improves over prior .5/.4 head; NE half barely improves over original.
backup273 preserves all seven current tensors. Need assess persistence and
record-supported rollout scale, not equate immediate decoder slopes with success.

## Forward24 / further independent factual validation
Current weaker broadened head (58b516f6e0), tuning stays200–299:
NLL1.61463 vs original1.61094, delta+.00369 over4563 readings.
Together with stays100–199 (+.00409), weighted independent delta~+.00389.
Current candidate remains saved and backed up in backup273.

## Run 275 / balanced broadened head staged
Verified current tensors against grid269(.25,.15) and arch263 rows.
Changed only final value-head weights/bias to grid269(.25,.25).
Offline deltas +.001181/+.004463/+.003732; immediate slopes
[-1.761,-1.829,+3.033,+1.981]. backup273 preserves the independently
validated .25/.15 candidate. Pending GPU factual and rollout tests.

## Run 276 / restart evidence recovery
Read architecture, record schema, early causal-identification notes, live fitting-object summaries, and recent audit stay IDs. No weights changed; balanced broadened .25/.25 remains staged pending GPU factual and paired-rollout validation.

## Run 277 / concise recovery after output truncation
Retrieved later identification notes and exact paired-audit IDs. Rejected early sign-reversed SSM estimates are not treated as truth. No weights changed.

## Forward 25–26 / balanced broadened head
arch263 + grid269(.25,.25): paired factual on tuning100–199 1.62067 vs 1.61582, delta +.00485 (4664 readings). Paired rollout [-10.008,+5.090,+2.209,-.928], original [-4.072,+2.273,+.615,-.858]. Relative to .25/.15, NE x2 improves modestly, insulin weakens, and factual margin narrows. No full causal repair claimed; backup273 remains available.

## Runs 279–280 / records-fitted bin calibration
Run279 stopped before fitting because nll228 requires bins. Run280 matched the cached observed-bin vector by reproducing the previously validated offline score, then fitted closed-form sensor-specific count offsets on train fitmask only. No weights saved; candidate scores retained in calgrid279.

## Run 281 / factor-balanced causal regressions
Rejected supervised bin calibration: no consistent validation gain. Reconstructed separate x2 and half-dose contrast statistics and fitted final heads analytically with extra half-dose weighting. No weights changed; weighted281 retains candidates, backup281 preserves balanced .25/.25.

## Run 282 / restart recovery
No weights changed. Recovering architecture, records-derived causal evidence, and weighted281 candidate structure; baseline reconstruction discrepancy must be understood before staging a reweighted head.

## Run 283 / bounded recovery
No weights changed. Previous output was truncated before candidate structures; using bounded diagnostics to recover rebalanced-head objects and later causal-identification evidence.

## Run 284 / regression audit
No weights changed. Inspecting weighted_fit281 scaling and contrast assembly before staging; accepted immediate effects come from corrected DML and insulin threshold checks, not the rejected early SSM fits.

## Run 285 / rebalanced candidate staged
All-five moment reconstruction errors: {'insulin': (1.1102230246251565e-16, 8.881784197001252e-16, 3.8020948115017603, 2.21253386375365), 'norepinephrine': (4.440892098500626e-16, 8.881784197001252e-16, 4.124591381987157, 3.3228599816546414)}. Candidate reproduction error 0. Baseline discrepancy reflects replacing five-factor NE constraints with x2/half constraints, not a failed numerical solve. Saved weighted281 (.3,.25,8) final value-head weights/bias; other five edited tensors unchanged. backup285 preserves balanced .25/.25. Offline NLL deltas +.000727/+.003526/+.002973. GPU factual and rollout validation pending; additional-dose generalization still needs auditing.

## Run 286 / multi-dose-anchored alternatives
Reproduced current offline scores. Prepared tuning stays100–199 for GPU audit. Fitted 24 unsaved closed-form heads with all-five NE-factor anchor .05 plus separate x2/half constraints; retained in grid286. Current saved tensors remain weighted281(.3,.25,8).

## Forward27–28 / half-balanced head validated
weighted281(.3,.25,8), hash fefaaf6371: tuning100–199 factual delta +.00386; paired rollout [-10.078,+4.988,+2.177,-1.133]. Insulin retained, NE half improved, NE x2 still too small. backup287 preserves this candidate.

## Run 287 / multi-dose-anchored stronger NE head staged
Saved grid286(.35,.3,8), all-five NE anchor .05, changing only final value weights/bias. Offline NLL deltas +.001193/+.004172/+.003575; immediate slopes [-1.803,-1.825,+2.774,+2.418]. Other five edited tensors unchanged. GPU audit pending.


## Runs 289–290 / current candidate audit and alternatives
Run289 stopped at Tensor.copy; no edits. grid286(.35,.3,8), hash df0778bbce: Forward29 factual +.00481 on tuning100–199; Forward30 rollout [-10.201,+4.921,+2.423,-1.151]. Forward31 factual +.00399 on tuning300–399; pooled factual delta about +.00440. No weights changed; backup290 preserves all current edits. Comparing closed-form alternatives and recovering quiet-channel construction before strengthening.


## Run 291 / stronger joint-dose head staged
Staged grid286(.45,.4,8), changing only final value-head weights and bias. Local dose slopes improve in all four contrasts relative to grid286(.35,.3,8); offline factual deltas +.001528/+.004597/+.004303. backup290 preserves the audited df0778bbce candidate. Pending paired GPU factual and rollout tests; local slope agreement is not proof of rollout agreement.

## Run 292 / factual-budget rollback
Forward32 (fwd_34), strong grid286(.45,.4,8), factual delta +.00560 on tuning100–199, exceeds +.005 on this cohort. Preserved strong tensors in backup292_strong and restored all seven backup290 tensors: audited grid286(.35,.3,8). Recovering records-based targets before final independent factual/rollout audits.

## Run 293 / bounded final-audit preparation
No weight changes. Recovered later causal notes, grid286 candidate metadata, and factual audit cohorts 0–99 and 200–299 from tuning.csv.

## Run 294 / restored-head verification
Restored final head matches grid286(.35,.3,8) to <1e-6. No weight changes. Prepared two remaining 100-stay factual cohorts. Provisional run23 kernel targets [-10.675,+5.338,+4.326,-2.163] are estimates, not observed counterfactual truth.

## Run 296 / NE-focused alternative staged
Run295 stopped on a missing artifact path before changing any weights. Preserved audited fallback in backup296. Staged grid286(.35,.5,4), changing only final value-head weights/bias; GPU audit pending. Recovered fallback factual files/results: {0: (4751, 0.003799494843190754, 'fwd_35'), 1: (4664, 0.00481093481989725, 'fwd_30'), 2: (4563, 0.004177602454525475, 'fwd_36'), 3: (4676, 0.003987617621899142, 'fwd_33')}.

## Run 297 / final-selection recovery
No weights changed. Preserved current NE-focused candidate in backup297_current; recovered architecture, schema, fallback structure, causal-note pointers, and all four tuning audit cohorts. Latest cohort factual delta is +.00515, so current candidate needs broader auditing and possibly interpolation toward backup296.

## Run 298 / reject NE-focused head; conservative joint-head test
Forward35, NE-focused hash a1da07951e: factual delta +.00515 on tuning100–199. Forward36: shifts [-10.002,+5.013,+2.482,-1.111], versus audited fallback [-10.201,+4.921,+2.423,-1.151]. The small NE-x2 gain does not justify worse factual fit and weaker NE-half. Restored backup296, then staged a final-head-only 80/20 fallback/backup292_strong blend: True. Other five edited tensors are unchanged. GPU factual and rollout audits are required before accepting this blend.

## Run 299 / final audit recovery
No weights changed. Current edit is the final-head-only 80/20 blend of backup296 and backup292_strong; other five edited tensors are unchanged. Forward41 (fwd_43), tuning cohort300–399, reported factual delta +0.00412 over4676 readings. Recovered architecture, record schema, identification notes, recent artifact scores, backup structure and tuning cohort IDs. One Python run remains, reserved for final selection and notebook completion.
Recovered factual artifact scores: {30: {'original': (4664, 1.615815973413379), 'edited': (4664, 1.6206269082332763), 'delta': 0.00481093481989725}, 33: {'original': (4676, 1.6355843883661247), 'edited': (4676, 1.6395720059880239), 'delta': 0.003987617621899142}, 34: {'original': (4664, 1.6158159948542024), 'edited': (4664, 1.6214155660377358), 'delta': 0.005599571183533492}, 35: {'original': (4751, 1.6316425173647653), 'edited': (4751, 1.635442012207956), 'delta': 0.003799494843190754}, 36: {'original': (4563, 1.6109350208196362), 'edited': (4563, 1.6151126232741617), 'delta': 0.004177602454525475}, 37: {'original': (4664, 1.6158159948542024), 'edited': (4664, 1.620971526586621), 'delta': 0.005155531732418561}, 39: {'original': (4664, 1.6158160377358488), 'edited': (4664, 1.6207749142367067), 'delta': 0.004958876500857912}, 41: {'original': (4751, 1.6316424752683645), 'edited': (4751, 1.6355978530835613), 'delta': 0.003955377815196837}, 42: {'original': (4563, 1.6109350646504494), 'edited': (4563, 1.615265439403901), 'delta': 0.004330374753451549}, 43: {'original': (4676, 1.6355844311377246), 'edited': (4676, 1.6397008340461934), 'delta': 0.004116402908468819}}

## Run 300 / final weight selection
Python budget exhausted. Selected: audited backup296 fallback, grid286(.35,.3,8).
Blend pooled factual audit: {'complete': True, 'n': 18654, 'delta': 0.004338372467031199, 'stays': 400, 'cluster_se': 0.0007944322498711077, 'upper95': 0.005645213518069171}
Fallback pooled factual audit: {'complete': True, 'n': 18654, 'delta': 0.0041920285193524165, 'stays': 400, 'cluster_se': 0.000780244353183863, 'upper95': 0.005475530480339871}
Selection requires blend pooled delta <=.00465 and its one-sided cluster-based 95% upper bound <=.005; otherwise restore audited fallback. This is a sampling-risk check, not a guarantee on unseen stays.
Selected existing paired rollout: {'insulin_half': 2.121263027715831, 'insulin_x2': -4.43856745969799, 'ne_half': -0.5724271764817991, 'ne_x2': 1.1981864255765198}
Original paired rollout: {'insulin_half': 0.965245896555745, 'insulin_x2': -1.7610999467255528, 'ne_half': -0.42788396702877834, 'ne_x2': 0.301541675242996}
Records support immediate insulin slope near -1.95, with independent threshold corroboration; corrected active-dose NE slope near +3.1 has weaker causal identification. Earlier opposite-sign SSM fits were rejected. Run23 rollout targets [-10.675,+5.338,+4.326,-2.163] remain provisional record-regression estimates, not observed counterfactual truth. Remaining forward calls will audit the fixed selected weights on the other tuning cohorts. No further weight edits are possible within budget. All-output loss and unseen intervention factors cannot be directly certified by the available forward readouts.
