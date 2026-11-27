# World-model weight repair

## Initial plan
- Infer drug effects exclusively from the supplied simulated records.
- Separate dose assignment from subsequent physiological response; do not assume either effect's sign or magnitude.
- Read architecture before choosing a weight-editing method.
- Use statistical fits and algebraic edits only; no model gradients or optimizers.
- Validate all-output factual fit as well as dose-rollout behavior, including proportionality and alternate measurement channels.
- Preserve the original model as the reference.

## Run 1
Read architecture, data documentation, vocabulary, tensor shapes, and record samples.
Counts: 1 Python run, 0 forward passes.

## Run 2
Architecture: doses enter action embeddings, z posterior, state attention, and last-rate registers. Rollouts feed generated measurements back into state. Loaded both record splits and inspected timing/dose variation. Run 1 output was truncated, so later inspection uses compact summaries.
Counts: 2 Python runs, 0 forward passes.

Run 3 formatting failure fixed: use plain text, not optional tabulate.
         event           drug      n     crude  adjusted  within_stay
glucose_finger        insulin 299023 -2.291109 -2.138886    -1.956919
   glucose_lab        insulin  28899 -0.486829 -1.605992    -1.615885
  map_arterial norepinephrine 314645  0.363832  1.102109     1.757063
      map_cuff norepinephrine 332093  0.550853 -2.990311     0.255970
## Forward 1: original dose rollout, 100 tuning stays
Insulin x2=-1.575, half=+0.910; norepinephrine x2=+0.178, half=-0.752. Norepinephrine arterial readout reaches only 13 stays in this batch.

## Run 4: cross-fitted dose-innovation diagnostics
Flexible dose/outcome regressions use only pre-infusion records, initial measurements, prior means, reading ages, and previous doses. Folds are separated by stay. Cluster standard errors use stays. Residual associations still require checking unmeasured confounding and outcome-history measurement noise.
         event           drug      subset      n  dose_residual_sd      beta  cluster_se
glucose_finger        insulin         all 299023          0.571650 -2.210592    0.045923
glucose_finger        insulin     treated 256645          0.615635 -2.206701    0.046053
glucose_finger        insulin hours_6plus 231375          0.579714 -2.256371    0.051238
   glucose_lab        insulin         all  28899          0.396308 -1.874827    0.123856
   glucose_lab        insulin     treated  11672          0.620436 -1.873376    0.124573
   glucose_lab        insulin hours_6plus  21716          0.406163 -1.974055    0.138091
  map_arterial norepinephrine         all 314645          0.092298  3.273393    0.090766
  map_arterial norepinephrine     treated 144316          0.136092  3.284740    0.090872
  map_arterial norepinephrine hours_6plus 235902          0.092019  3.481080    0.103081
      map_cuff norepinephrine         all 332093          0.060543  2.584580    0.224327
      map_cuff norepinephrine     treated  73973          0.127912  2.671945    0.224919
      map_cuff norepinephrine hours_6plus 248989          0.060424  3.613785    0.256323
Counts: 4 Python runs, 1 forward passes.

## Forward 2 / Run 5
Exported original states for training stays 1010000000..1010000099 (all bundles). Checked held-out-stay dose decoding and residual-effect dose-range diagnostics. Past-change placebo here is raw, not separately residualized, and must not be treated as a definitive independence test.
         event                range      n      beta       se
glucose_finger               0.5..1  21312 -1.895856 0.144475
glucose_finger                 1..2 134998 -2.042501 0.083923
glucose_finger              2..14.5 100335 -2.306254 0.059658
glucose_finger prior_change_placebo 256186 -0.307860      NaN
   glucose_lab               0.5..1   1099 -1.552074 0.390301
   glucose_lab                 1..2   6440 -1.674272 0.221519
   glucose_lab               2..9.5   4133 -1.994494 0.159362
   glucose_lab prior_change_placebo   5781  0.010951      NaN
  map_arterial           0.01..0.31  46952  2.808803 0.267351
  map_arterial            0.31..0.7  48870  3.186511 0.130080
  map_arterial               0.7..1  48494  3.503649 0.137606
  map_arterial prior_change_placebo 298800  0.134794      NaN
      map_cuff           0.01..0.21  24183  1.573456 0.905161
      map_cuff           0.21..0.62  25354  2.622023 0.337926
      map_cuff              0.62..1  24436  2.837563 0.321496
      map_cuff prior_change_placebo 279926  0.482887      NaN
 features  alpha           drug  test_r2  test_rmse
      out    1.0        insulin 0.739124   0.663240
      out    1.0 norepinephrine 0.790876   0.158897
      out   30.0        insulin 0.690299   0.722644
      out   30.0 norepinephrine 0.743011   0.176145
      out  300.0        insulin 0.652943   0.764986
      out  300.0 norepinephrine 0.697538   0.191095
out_and_z    1.0        insulin 0.899701   0.411246
out_and_z    1.0 norepinephrine 0.795638   0.157077
out_and_z   30.0        insulin 0.890078   0.430523
out_and_z   30.0 norepinephrine 0.777885   0.163758
out_and_z  300.0        insulin 0.830641   0.534389
out_and_z  300.0 norepinephrine 0.732674   0.179653
all_state    1.0        insulin 0.899256   0.412158
all_state    1.0 norepinephrine 0.779078   0.163318
all_state   30.0        insulin 0.892723   0.425312
all_state   30.0 norepinephrine 0.781131   0.162557
all_state  300.0        insulin 0.838499   0.521844
all_state  300.0 norepinephrine 0.748401   0.174288
Counts: 5 Python runs, 2 forward passes.

## Forward 3 / Run 6
Original factual NLL on matching 100 training stays: 1.63065 (4745 values). Reconstructing prediction heads from exported states; compared candidate activations, gap encodings, and last-register layouts. No edits yet.
                                             config  mean_rmse  nll_rmse      nll
('silu', 'log1p', 'gelu', 'interleaved', 0, 'silu')   4.519133  0.451487 1.712594
  ('silu', 'log', 'gelu', 'interleaved', 0, 'silu')   4.519134  0.451485 1.712592
  ('silu', 'log', 'gelu', 'interleaved', 0, 'gelu')   4.559348  0.425870 1.703790
('silu', 'log1p', 'gelu', 'interleaved', 0, 'gelu')   4.559350  0.425875 1.703792
  ('gelu', 'log', 'gelu', 'interleaved', 0, 'silu')   4.613220  0.477999 1.724570
('gelu', 'log1p', 'gelu', 'interleaved', 0, 'silu')   4.613243  0.478005 1.724573
  ('gelu', 'log', 'silu', 'interleaved', 0, 'silu')   4.701380  0.467703 1.724681
('gelu', 'log1p', 'silu', 'interleaved', 0, 'silu')   4.701406  0.467708 1.724684
Counts: 6 Python runs, 3 forward passes.

## Run 7
Re-reading architecture and inspecting persistent variables before a targeted algebraic edit. Reconstruction remains too inaccurate to use safely; no tensors edited.
Counts: 7 Python runs, 3 forward passes.

## Run 10
Direct reconstruction grid over gap scale, activation, latent source, query type, last-register flags and status. Still no edits.
[(('silu', 0.1, 'gelu', 'prior', 1, 3, 0, 'silu'), 4.607177356790052, 0.4679384991024623, 1.7100245952606201), (('silu', 0.1, 'silu', 'zero', 1, 3, 0, 'silu'), 4.61602379769742, 0.47527918777192124, 1.7125592231750488), (('silu', 0.1, 'gelu', 'zero', 1, 3, 0, 'silu'), 4.61602379769742, 0.47527918777192124, 1.7125592231750488), (('silu', 0.1, 'gelu', 'prior', 1, 3, 0, 'gelu'), 4.666139433119243, 0.4374716970171015, 1.7025305032730103), (('silu', 0.1, 'silu', 'zero', 1, 3, 0, 'gelu'), 4.6671216267476865, 0.4484234491205614, 1.7064038515090942)]
Counts: 10 Python runs, 3 forward passes.

## Run 11
Checking latent-probability averaging instead of a deterministic latent mean.
[(('silu', 'gelu', 'gelu', 1.0), 4.599382696680694, 0.43500487927887543, 1.706592082977295, 0.44631290459442136, 0.06186679615491629), (('silu', 'gelu', 'gelu', 0.5), 4.599914960000067, 0.4363669175661295, 1.7085585594177246, 0.424250216720581, 0.06383318363541365), (('silu', 'gelu', 'silu', 1.0), 4.65668039098584, 0.4705122763196616, 1.7173858880996704, 0.5321874763946532, 0.0726605230089426), (('silu', 'gelu', 'silu', 0.5), 4.6756450136965455, 0.4724108044289337, 1.7201770544052124, 0.5108177330474853, 0.07545164412671328)]
Counts: 11 Python runs, 3 forward passes.

## Run 13
Best reconstruction now has mean RMSE 1.633 and NLL RMSE 0.245, still not exact. Re-reading architecture, shapes, and persistent helper inventory before choosing a directly validated algebraic weight edit. No edits yet.
Counts: 13 Python runs, 3 forward passes.

## Run 15
Inspected panel indexing, examples, and saved dose-innovation residuals. Next priority is causal dynamics and properly residualized pre-dose placebos, not more uncontrolled head-reconstruction grids.
Counts: 15 Python runs, 3 forward passes.

## Runs 16–17: six-hour local projections
Run 16 failed because exposure indices extended one hour too far; corrected in run 17. No edits.
Two-fold stay-separated HGB adjustment using pre-dose history, current main reading observed. Deconvolution separates later policy-dose changes from an estimated fixed-dose response. Requires conditional exogeneity; not established merely by this regression.
('glucose_finger', 'insulin') y_response=[(-1.850340141162819, 0.05921533919140744), (-2.1762604567431203, 0.06489918613459421), (-1.7410683131830231, 0.06761284716241134), (-1.2531025010849781, 0.06920144163811291), (-0.6551003687865006, 0.07183981699949399), (-0.3467783443919157, 0.07189773446145925)] dose_response=[(1.0, 0.0), (-0.043158439043129916, 0.00367638110517134), (-0.05631395190790126, 0.0038729998584322244), (-0.05248995784797088, 0.0038764553121176304), (-0.03888142649128522, 0.004054276323950561), (-0.011719974297229152, 0.004260511826010613)] kernel=[-1.85034014 -2.25611825 -1.94263882 -1.56111897 -1.02224091 -0.69018551]
('map_arterial', 'norepinephrine') y_response=[(3.3505774844819856, 0.11296414420802602), (5.0683432874472025, 0.1299952890881905), (5.506229788027758, 0.1425291933163007), (5.131865472690208, 0.1518675987200649), (4.4902343523316315, 0.15979000044670347), (3.944962670038462, 0.16692614695654193)] dose_response=[(1.0, 0.0), (0.7304205725783163, 0.004233601628482961), (0.5594909967595264, 0.0052650548567893035), (0.43833097967373025, 0.005683985328155739), (0.3576708447723065, 0.005847420576034897), (0.2858912553960495, 0.006022535310425034)] kernel=[3.35057748 2.62101256 1.71717035 0.94251408 0.49378644 0.40891321]
Counts: 17 Python runs, 3 forward passes.

## Run 18: causal robustness
Removed early-mean current-dose leakage at exposure hour 3; richer HGB adjustment and held-out simultaneous pre-dose assay diagnostics. A held-out assay association is a confounding diagnostic, not definitive proof, since clinicians may also use that assay.
('glucose_finger', 'insulin') beta=-1.920236160942765 se=0.057549713623945384 withheld-assay=(0.12194404441190956, 0.07962629954886274)
('map_arterial', 'norepinephrine') beta=3.3338848273633017 se=0.11219104588336076 withheld-assay=(-0.09941991398673201, 0.28240422270011895)
Selected 100 representative training stays enriched for both measured drug pathways. No edits.
Counts: 18 Python runs, 3 forward passes.

## Restart / Run 19
Forward 4 exported 5,215 original bundle states on representative training stays. No weights edited. Rechecking architecture and persistent analysis objects before a directly validated algebraic correction; treatment conclusions remain conditional on pre-dose-history adjustment.
Counts: 19 Python runs, 4 forward passes.

## Run 20
Architecture confirms distinct action encoder and shared numeric/bin embeddings. Retrieving retained analysis objects and original intervention predictions. No edits.
Counts: 20 Python runs, 4 forward passes.

## Run 21
Recovered existing hourly panels, cross-fitted effect diagnostics, representative state sample, and original rollout table. Retrieving quantitative baseline and normalization before constructing a reversible edit. No edits.
Counts: 21 Python runs, 4 forward passes.

## Runs 22–23
Run 22 had a syntax error and executed nothing. Run 23 checks exposed-readout averaging and event timing and selects an enriched tuning intervention panel. No edits.
Counts: 23 Python runs, 4 forward passes.

## Forward 5 / Run 24
Enriched original tuning rollout: insulin x2=-1.773, half=+0.816 (665 exposed hours); NE x2=+0.048, half=-0.481 (571 exposed hours). Exploring innovation-decoding in value-head hidden features to preserve teacher-forced non-value outputs exactly. No edits.
Counts: 24 Python runs, 5 forward passes.

## Run 25
Recovered architecture, earlier notebook and retained-object inventory. No edits; approximate head reconstruction is not yet a factual-fit certificate.

## Forward 6 / Run 29
Original enriched training factual NLL 1.51284. Built value-head calibration features and stay-separated innovation-decoder tests; no edits yet. Combined only stays shared by state and factual exports.

## Run 30
Innovation decoding from reconstructed value-head features failed stay-separated validation (negative held-out R2 for both drugs). No edit accepted. Re-inspecting architecture, tensor shapes, normalization and retained objects to design a direct, reversible action-pathway correction. Counts: 30 Python runs, 6 forward passes.

## Runs 31–33
Inspected normalized dose units and pure-action timing. Infusions follow hourly readings by five minutes; subsequent drug bundles include END atoms. Testing posterior reconstruction against exported original latent means before any posterior-output edit.
Best posterior reconstruction candidates: [(0.139067143201828, 'none', (), True, 'left', 'relu', 'silu', 0.1, 'out_plus_flow', 'relu'), (0.13929472863674164, 'none', ('status', 'route', 'action'), True, 'left', 'relu', 'silu', 0.1, 'out_plus_flow', 'relu'), (0.13951130211353302, 'none', ('route', 'action'), True, 'left', 'relu', 'silu', 0.1, 'out_plus_flow', 'relu')]
Counts: 33 Python runs, 6 forward passes. No edits.

## Run 34
Tested empty observation-encoder output, END routing, posterior layouts and prior residual parameterization against exported latent means.
Best: [(0.02039741910994053, ('none', 'left', 'gelu', 'starts_action_ends_observation', False, 'gelu', 'out_plus_flow', 'gelu', 'first', 'direct')), (0.023720484226942062, ('none', 'left', 'gelu', 'starts_action_ends_observation', False, 'silu', 'out_plus_flow', 'gelu', 'first', 'direct')), (0.02458619512617588, ('none', 'left', 'gelu', 'starts_action_ends_observation', False, 'relu', 'out_plus_flow', 'gelu', 'first', 'direct'))]
Counts: 34 Python runs, 6 forward passes. No edits.

## Run 35
END atoms enter the observation encoder; start atoms enter the action encoder. GELU and nonzero empty observation encoding are strongly supported by posterior reconstruction. Checked END metadata and pooling counts.
Best: [(0.018507156521081924, (2, False, False, True, False, False, False)), (0.01863560453057289, (0, False, True, True, False, False, False)), (0.0186410304158926, (0, False, False, True, False, False, False))]
Counts: 35 Python runs, 6 forward passes. No edits.

## Restart, runs 36–39
Run 36 reconstruction failed on tuple/dict indexing; posterior reconstruction remained unreliable outside action bundles. Run 37 read exact architecture. Run 38 inventory failed on a module's callable shape attribute. Run 39 uses explicit type checks to recover persistent analyses. No weights changed; 39 Python runs, 6 forward passes.

## Runs 42–43: corrected posterior innovation decoding
Run 42 assertion caught tuple layout mismatch. Correct layout: (stay index, exposure hour, dose, dose residual, outcome residual, predicted dose, predicted outcome). No edits.
[{'feature': 'hidden', 'alpha': 1.0, 'drug': 'insulin', 'active_n': 675, 'active_r2': 0.21516390626509652, 'active_rmse': 0.5312344198187235, 'nonaction_rms': 0.05093352851850605}, {'feature': 'hidden', 'alpha': 1.0, 'drug': 'norepinephrine', 'active_n': 329, 'active_r2': -0.008181327097222413, 'active_rmse': 0.12298249435648999, 'nonaction_rms': 0.005553259526670194}, {'feature': 'hidden', 'alpha': 10.0, 'drug': 'insulin', 'active_n': 675, 'active_r2': 0.12239428463434576, 'active_rmse': 0.561754349627384, 'nonaction_rms': 0.03207191579397678}, {'feature': 'hidden', 'alpha': 10.0, 'drug': 'norepinephrine', 'active_n': 329, 'active_r2': -0.0014827990903116817, 'active_rmse': 0.12257325516755867, 'nonaction_rms': 0.005463907953841229}, {'feature': 'hidden', 'alpha': 100.0, 'drug': 'insulin', 'active_n': 675, 'active_r2': 0.07323191414784547, 'active_rmse': 0.5772743452668887, 'nonaction_rms': 0.030996514161188305}, {'feature': 'hidden', 'alpha': 100.0, 'drug': 'norepinephrine', 'active_n': 329, 'active_r2': 0.0005148659203935146, 'active_rmse': 0.12245094526298009, 'nonaction_rms': 0.0072987249431671975}, {'feature': 'input', 'alpha': 1.0, 'drug': 'insulin', 'active_n': 675, 'active_r2': 0.6488263629322728, 'active_rmse': 0.3553511090004981, 'nonaction_rms': 0.05469963678194881}, {'feature': 'input', 'alpha': 1.0, 'drug': 'norepinephrine', 'active_n': 329, 'active_r2': 0.048216887067118464, 'active_rmse': 0.11949313904315578, 'nonaction_rms': 0.005768533997113083}, {'feature': 'input', 'alpha': 10.0, 'drug': 'insulin', 'active_n': 675, 'active_r2': 0.5139871783063549, 'active_rmse': 0.4180426438629263, 'nonaction_rms': 0.060417761856946715}, {'feature': 'input', 'alpha': 10.0, 'drug': 'norepinephrine', 'active_n': 329, 'active_r2': 0.020784603090882148, 'active_rmse': 0.12120292178921457, 'nonaction_rms': 0.005902958854987414}, {'feature': 'input', 'alpha': 100.0, 'drug': 'insulin', 'active_n': 675, 'active_r2': 0.38338809109241445, 'active_rmse': 0.4708717942225751, 'nonaction_rms': 0.0655938157834504}, {'feature': 'input', 'alpha': 100.0, 'drug': 'norepinephrine', 'active_n': 329, 'active_r2': 0.01506820657199437, 'active_rmse': 0.12155618203402276, 'nonaction_rms': 0.005830448220832881}]

## Restart / Run 45
Run 44 stopped on empty action pooling, before a validated edit. Recovering architecture and retained objects to fix empty-bundle handling. No new causal conclusion or factual-fit claim follows from that error.
Counts: 45 Python runs, 6 forward passes.

## Run 46
Architecture requires zero action encoding for empty bundles. Run 45 output was truncated; inspecting the helper and retained objects compactly. No weights changed. Counts: 46 Python runs, 6 forward passes.

## Run 47
The failed helper's entire action-atom collection was empty. Checking action/state alignment before trusting reconstruction or decoder tests. No edit applied. Counts: 47 Python runs, 6 forward passes.

## Run 48
Rebuilt action atoms using fractional bundle-hour keys. Tested stay-held-out innovation decoding and dose response at x0, x0.5, x1.5, x2 and x3. No model weights edited.
('insulin', 0.0, 'input', [0.6804, -0.0005])
('insulin', 0.0, 'hidden', [0.3188, 0.0056])
('insulin', 0.5, 'input', [0.7649, 0.0082])
('insulin', 0.5, 'hidden', [0.3688, 0.0106])
('insulin', 1.5, 'input', [0.4913, 0.0047])
('insulin', 1.5, 'hidden', [0.1625, 0.0064])
('insulin', 2.0, 'input', [0.4741, 0.0038])
('insulin', 2.0, 'hidden', [0.1468, 0.0065])
('insulin', 3.0, 'input', [0.3451, 0.0021])
('insulin', 3.0, 'hidden', [0.0978, 0.0042])
('norepinephrine', 0.0, 'input', [-0.2399, 0.0683])
('norepinephrine', 0.0, 'hidden', [-0.1252, -0.0855])
('norepinephrine', 0.5, 'input', [0.165, 0.1849])
('norepinephrine', 0.5, 'hidden', [0.3387, 0.9239])
('norepinephrine', 1.5, 'input', [0.6413, 0.0798])
('norepinephrine', 1.5, 'hidden', [0.0677, 0.2844])
('norepinephrine', 2.0, 'input', [0.2845, 0.0738])
('norepinephrine', 2.0, 'hidden', [-0.2992, 0.1667])
('norepinephrine', 3.0, 'input', [0.7893, 0.0171])
('norepinephrine', 3.0, 'hidden', [-0.087, 0.1085])
Counts: 48 Python runs, 6 forward passes.

## Run 49
Original innovation decoders attenuate and distort out-of-range dose changes. Fitted small ridge decoders with explicit physical-dose-difference constraints across x0, x0.5, x1.5, x2 and x3, plus drug-presence gates. These are statistical fits, not optimisation of model weights. No edit yet.
Counts: 49 Python runs, 6 forward passes.

## Run 50
Constrained original-feature decoders were rejected for norepinephrine inaccuracy and cross-drug leakage. Computed a candidate rank-four action-output edit carrying transparent drug rates and presence. Directions are orthogonal to constant components and have minimal original variance. Candidate remains in memory, not submitted as weights.
(('base', 1.0), [0.018743156282418343, 0.03856396780209878], 0.044145389715116945)
((0, 0.0), [0.28170793280870776, 0.06188839684516779], 0.04453734565833278)
((0, 0.5), [0.07908778429361159, 0.06403630709943581], 0.05508879506799491)
((0, 1.5), [0.015168043100967494, 0.03642655975126325], 0.03705072054437619)
((0, 2.0), [0.009878892872640928, 0.03187586230692108], 0.027081144884467778)
((0, 3.0), [0.05417645864787989, 0.028811501586909462], 0.027780991286927237)
((1, 0.0), [0.008928583762445512, 0.1191617952643196], 0.024026556174797764)
((1, 0.5), [0.01923440106732315, 0.08963056106144593], 0.049609895423881056)
((1, 1.5), [0.017306205612216193, 0.02470510037285471], 0.023338974368020127)
((1, 2.0), [0.019872597582844164, 0.018000576058170466], 0.028570984875955108)
((1, 3.0), [0.024157133036691376, 0.015671960166781778], 0.039134222028528565)
Counts: 50 Python runs, 6 forward passes.

## Run 51
Refined the transparent rate/presence trace regression and fitted expected-dose offsets using only pre-action ctx and observation encoding, separately for each drug. Drug innovations will be gated to action bundles; dose changes do not enter the expected-dose offset directly. All fits are closed-form statistical regressions. No weights saved yet.
Selected trace ridge: 0.0001; expected-dose ridges: {'insulin': 0.1, 'norepinephrine': 1.0}.
Counts: 51 Python runs, 6 forward passes.

## Run 52: restart recovery
Run 51 recovered insulin innovations but not norepinephrine innovations. No edit is validated or applied. Recovering causal evidence, architecture, and retained fits.
Counts: 52 Python runs, 6 forward passes. No edits.

## Run 53
Run 52 output was truncated before retained-object inventory. Recovering later causal checks and candidate fits compactly. No edit.
Counts: 53 Python runs, 6 forward passes.

## Run 54
Recovered multi-hour dose-response estimates. Retrieving remaining confounding checks and recent candidate matrices without broad inventories.
Counts: 54 Python runs, 6 forward passes. No edits.

## Run 55
Later robustness estimates: insulin one-hour beta -1.9202, NE +3.3339; withheld-assay diagnostics were compatible with zero. Recent objects did not use guessed r48_...r51_ prefixes; locating via saved inventory.
Counts: 55 Python runs, 6 forward passes. No edits.

## Run 56
Located rr50/rr51 trace and expected-dose candidates. Inspecting dimensions and downstream paths before any reversible weight trial.
Counts: 56 Python runs, 6 forward passes. No edits.

## Run 57
Inspecting exact action cache layouts and low-impact posterior neurons. Expected-dose offsets require gating; NE offset accuracy remains weak. No saved edit.
Counts: 57 Python runs, 6 forward passes.

## Run 58
Built selected trace candidate and its linear reader; checked held-out extrapolation. Estimated posterior latent reading directions by finite record-value perturbations, no gradients. No tensors saved. NE expected-dose predictor needs physical bounds.
Counts: 58 Python runs, 6 forward passes.

## Run 59
Recovered architecture, early causal evidence and recent candidate objects. Run 58's trace is promising but measurement reconstruction error remains unresolved; no edit saved. Counts: 59 Python runs, 6 forward passes.

## Run 60
Run 59 output was truncated. Recovering later causal checks and candidate dimensions without repeating architecture. No weights changed. Counts: 60 Python runs, 6 forward passes.

## Run 61
Recovered conditional one-hour effects (-1.9202 insulin, +3.3339 NE) and six-hour deconvolved kernels. Inspecting expected-dose fits and observation helper before a residual-based posterior edit. No edits. Counts: 61 Python runs, 6 forward passes.

## Runs 63-64
Run 63 stopped on NumPy/Torch type mismatch. Run 64 converts cached arrays before checking alignment and fit layout. No weights changed. Counts: 64 Python runs, 6 forward passes.

## Run 65
Testing atom-vector agreement, encoder implementation and posterior reconstruction. Checking bounded expected-dose errors on held-out stays. No edits. Counts: 65 Python runs, 6 forward passes.

## Run 66
Run 65 showed cached and reconstructed bundles have different atom counts. Latent direction calibration must be repeated with verified stay-hour alignment. Unbounded NE expected-dose fit has large extrapolation error and is rejected. No weights changed. Counts: 66 Python runs, 6 forward passes.

## Run 67
Bundle identities match, but atom-count disagreement invalidates the latest latent calibration. Recovering architecture and original causal evidence before proceeding. Unbounded norepinephrine expected-dose fit remains rejected. No weights changed.
Counts: 67 Python runs, 6 forward passes.

## Run 68
Recovered initial conditional effect evidence and original rollout weakness. Retrieving multi-hour evidence and exact candidate dimensions with bounded output. No weights changed. Counts: 68 Python runs, 6 forward passes.

## Run 69
Recovered six-hour kernels: insulin [-1.850,-2.256,-1.943,-1.561,-1.022,-0.690]; NE [3.351,2.621,1.717,0.943,0.494,0.409]. Conditional-exogeneity assumptions remain necessary. Inspecting retained candidates without repeated evidence dumps. No edits.
Counts: 69 Python runs, 6 forward passes.

## Runs 70-71
Run 70 stopped on DataFrame column/row indexing; no validation claim follows. Run 71 fixes indexing and checks independent observation encoding and posterior agreement. No weights changed. Counts: 71 Python runs, 6 forward passes.

## Run 72
Run 71 found GELU drug posterior reconstruction mean error 0.003869, but observation calibration mixed integer reading times with fractional action bundles. Checking raw exported metadata and recalibrating using exact integer bundle times. No weights edited. Counts: 72 Python runs, 6 forward passes.

## Run 73
Run 72 used incompatible export/cache ordering; its directions are invalid and rejected. Aligning cached rows to exports by exact 32-dimensional posterior vectors and checking previous states. No weights changed. Counts: 73 Python runs, 6 forward passes.

## Run 74
Run 73 exactly aligned all 8443 cached posterior vectors and all previous states. Cache stay IDs were correct; cached hours were floored. Integer observation encodings match exactly. Posterior mismatch remains; testing true-time context and empty-action handling. No edits. Counts: 74 Python runs, 6 forward passes.

## Run 75
Reading-posterior reconstruction remains invalid. Recovering exact architecture and retained fits; no edits. Counts: 75 Python runs, 6 forward passes.

## Run 76
Architecture recovered; run 75 output was truncated before inventory. Inspecting retained candidate objects and tensor dimensions compactly. No edits. Counts: 76 Python runs, 6 forward passes.

## Run 77
No edit is applied. Narrowing retained-object inspection and recovering encoder helper implementations to diagnose posterior mismatch. Counts: 77 Python runs, 6 forward passes.

## Run 78
Action traces recovered. Testing whether END-atom encoder assignment explains reading-only posterior mismatch; no edit. Counts: 78 Python runs, 6 forward passes.

## Run 79
Controlled fixed-weight grid over END routing, empty encodings, and nonlinearities. No edits. Results retained in r79_results. Counts: 79 Python runs, 6 forward passes.

## Runs 80-81
All 8443 cached bundle counts match exported counts exactly. END-only z posteriors match exactly; drug-only median/max per-row RMSE 0.00277/0.04259; reading median 0.13633. END routing is correct. Suspect measurement categorical embedding fields, not bundle alignment. Run 80 stopped while inspecting tuple-valued expected-dose fits. No edits. Counts: 81 Python runs, 6 forward passes.

## Run 82
Restart recovery: verified prior evidence of exact bundle alignment and END-only posterior agreement. Reading posterior reconstruction is still invalid; checking architecture, vocabulary and actual categorical fields before calibration. No weights changed. Counts: 82 Python runs, 6 forward passes.

## Runs 83–84
Run 83 stopped while treating the list-valued numeric vocabulary as a dictionary. No experiment or weight edit occurred. Run 84 corrects inspection and recovers encoder inputs. Counts: 84 Python runs, 6 forward passes.

## Run 85
Run 84 recovered core helpers but output was truncated by binary lookup keys. Restricting inspection to encoder dependencies and atom metadata. No weights changed. Counts: 85 Python runs, 6 forward passes.

## Run 86
Testing measurement categorical unit IDs against aligned exported posteriors. This changes only local reconstruction arrays, not weights. Counts: 86 Python runs, 6 forward passes.
Unit-grid results: [(0.03494873270392418, 1, 2), (0.03506092354655266, 0, 2), (0.036011286079883575, 3, 2), (0.0433625765144825, 4, 2), (0.09017805755138397, 5, 2), (0.10899165272712708, 0, 1)]

## Run 87
Unit IDs alone reduced but did not eliminate sensor posterior error. Tested unit IDs jointly with bin-edge equality conventions. Best combinations: [(0.03494873270392418, 1, 2, 0, 0), (0.03506092354655266, 0, 2, 0, 0), (0.036011286079883575, 3, 2, 0, 0), (0.03636328503489494, 1, 2, 1, 0)]. Full-sensor row-error quantiles: [7.947202362856842e-08, 0.0013397112488746643, 0.0720740407705307, 0.14990916848182678]. No weights changed. Counts: 87 Python runs, 6 forward passes.

## Run 88
Bin-edge alternatives did not resolve residual error. Compared contexts and tested unit IDs independently by sensor. Context results: [(0.03494873270392418, 'core.out_norm/gap/gelu'), (0.03494873270392418, 'r80-input'), (0.034948837012052536, 'cached'), (0.03899077698588371, 'core.out_norm/gap/silu')]; selected units: {'glucose_finger': 0, 'glucose_lab': 5, 'map_arterial': 2, 'map_cuff': 2}; sensor median/90th/max errors: {'glucose_finger': [2.635600537814753e-07, 4.219517677483964e-07, 8.356386729246879e-07], 'glucose_lab': [2.608026079542469e-07, 4.142876264268125e-07, 7.331348115258152e-07], 'map_arterial': [2.605811459943652e-07, 4.134151652124274e-07, 8.356386729246879e-07], 'map_cuff': [2.5597904595997534e-07, 4.060261460381298e-07, 8.356386729246879e-07]}. No weights changed. Counts: 88 Python runs, 6 forward passes.

## Run 89
Reading reconstruction is exact with units finger=0, lab=5, both MAP=2 and left-edge bins. Tested additional drug dose-unit fields; best results: [(2.201097402121377e-07, 3, 4), (0.003081782255321741, 5, 4), (0.003580405144020915, 1, 4), (0.00379272922873497, -1, 4), (0.00379272922873497, 0, 4)]. Recovered retained candidates; no weights changed. Counts: 89 Python runs, 6 forward passes.

## Run 90
Drug and measurement encoder reconstruction is now exact after correcting categorical units. Recovering architecture and retained trace candidates before any weight edit. No tensors edited. Counts: 90 Python runs, 6 forward passes.

## Run 91
Architecture recovered: action encoder and z posterior both mediate dose effects. Recovering original causal evidence and trace targets before selecting weight surgery. No tensors edited. Counts: 91 Python runs, 6 forward passes.

## Run 92
Run 91 output truncated before candidate details; retrieving only missing compact sections. Record-based effect signs remain insulin negative and NE positive, conditional on pre-dose adjustment. No tensors edited. Counts: 92 Python runs, 6 forward passes.

## Run 93
Retained trace decoders have very large coefficient norms; require recalibration and extrapolation checks with corrected encodings before reuse. Inspecting posterior/action tensor layout and candidate construction. No tensors edited. Counts: 93 Python runs, 6 forward passes.

## Run 94
Recovering exact corrected atom construction and retained latent perturbation arrays. Transparent action-subspace approach must be rebuilt with correct units, not reused unchanged. No tensors edited. Counts: 94 Python runs, 6 forward passes.

## Runs 96–97
Run 96 exposed a retained state table from a different export than its selected index and stopped. Rejecting mixed global caches; loading state archives directly and isolating original weights/vocabulary. No tensors edited. Counts: 97 Python runs, 6 forward passes.

## Run 98
No weights edited. Retained r33 atom helper has obsolete unit IDs; do not reuse it uncorrected. Reading architecture before rebuilding a coherent calibration from original state exports.
Counts: 98 Python runs, 6 forward passes.

## Run 99
Architecture confirms distinct action-encoder and posterior dose pathways. Recovering record schema and earlier identification evidence before choosing a weight intervention. No weights edited.
Counts: 99 Python runs, 6 forward passes.

## Run 100
Recovered record-based dose-innovation evidence and original weak rollouts. These estimates require sequential-exogeneity assumptions. Created an isolated original-weight dictionary; inspecting tensor layout and prior reconstruction checks before deriving a new residualized edit. No edits applied.
Counts: 100 Python runs, 6 forward passes.

## Run 101
Fresh training/tuning records and state exports loaded separately. Prior six-hour deconvolved kernels support dynamic—not constant-offset—repair. Inspecting corrected helper interfaces and previous latent candidates; none will be reused without alignment and extrapolation checks. No edits.
Counts: 101 Python runs, 6 forward passes.

## Run 102
Fresh atom/bundle reconstruction with verified unit IDs. Export4 atom-count disagreements: 0 / 5215. Rejected obsolete innovation decoders because their dose extrapolation was distorted. No model weights edited.
Counts: 102 Python runs, 6 forward passes.

## Run 103
Posterior reconstruction best: (0.0636540874838829, ('drug_END_observation', False), ('rounded_minutes', 'log', 'gelu')). Valid-row maximum posterior RMSE: 0.42946339. Fresh records and exported state rows have matching atom counts. No weights edited.
Counts: 103 Python runs, 6 forward passes.

## Run 104
Fresh reconstruction gives numerical agreement for active-drug posteriors (max RMSE 5.5e-7), but not reading posteriors. Drug/time paths validated; observation path remains invalid. Checking atom construction and error dependence on bundle size. No edits.
Counts: 104 Python runs, 6 forward passes.

## Run 105
Dose-only reconstruction is validated, but the fresh observation path is not. Inspecting architecture and retained exact helpers before any calibration or weight edit. No edits. Counts: 105 Python runs, 6 forward passes.

## Run 106
Architecture recovered. Run 105 inventory was truncated by tensor representations; restricting output to fresh helper signatures and array shapes. No edits. Counts: 106 Python runs, 6 forward passes.

## Run 107
Recent objects use different names than expected. Locating actual kernel objects and recovering original causal findings; no edits. Counts: 107 Python runs, 6 forward passes.

## Run 108
Located fresh objects under rr_*. Original record evidence supports insulin-negative and NE-positive adjusted effects, subject to identification assumptions. Comparing fresh versus validated set encoders; no edits. Counts: 108 Python runs, 6 forward passes.

## Run 111
Correcting tensor-only encoder return handling and comparing full-range atoms, normalization constants, and best-context posterior. No edits. Counts: 111 Python runs, 6 forward passes.

## Run 112
Testing categorical-offset explanations against exported posteriors with an explicit cache-free set encoder. No weights edited. Counts: 112 Python runs, 6 forward passes.

## Run 113
Recovered architecture, identification notes, and helper inventory after restart. Run 112 ruled out rate-unit alternatives; reading reconstruction remains unvalidated. No weights edited. Counts: 113 Python runs, 6 forward passes.

## Run 114
Recovered early identification evidence and tensor layouts. No causal magnitudes inferred from signs alone; no weights edited. Counts: 114 Python runs, 6 forward passes.

## Run 115
Localized value-head editing is structurally possible, but dynamic effect and exact feature calibration are required. Recovering prior validated helpers rather than repeating broad reconstruction searches. No edits. Counts: 115 Python runs, 6 forward passes.

## Run 116
Prior deconvolved kernels imply multi-hour, decaying responses. Prioritizing a value-head-only algebraic edit to preserve other teacher-forced outputs. Checking atom construction and validated cache dependencies; no weights edited. Counts: 116 Python runs, 6 forward passes.

## Run 117
Comparing fresh and validated components only after alignment by exported posterior vectors. This isolates reconstruction errors from cache-order errors. No edits. Counts: 117 Python runs, 6 forward passes.

## Run 118
Aligned fresh context, observation, and action inputs agree with validated arrays. Recomputed posterior from these components to distinguish stale inputs from encoder defects. No edits. Counts: 118 Python runs, 6 forward passes.

## Run 119
Fresh posterior remains exact on drug bundles but not reading bundles. Directly testing empty-action handling. Rejecting the prior value-head innovation decoder because held-out R2 was negative; considering a gated action/posterior repair instead. No edits. Counts: 119 Python runs, 6 forward passes.


## Run 120
Run 119 resolved posterior reconstruction: empty actions must use the normalized empty-set encoding. Reading and drug posteriors now agree with exported states to numerical precision. Recovering architecture and quantitative effect estimates before constructing a fresh, reversible edit. No weights edited. Counts: 120 Python runs, 6 forward passes.


## Run 121
Recovered six-hour adjusted/deconvolved response kernels: insulin [-1.850,-2.256,-1.943,-1.561,-1.022,-0.690], NE [3.351,2.621,1.717,0.943,0.494,0.409]. These are record-based estimates under conditional exogeneity, not known simulator truth. Inspecting compact replay/head interfaces to construct a localized algebraic repair. No edits. Counts: 121 Python runs, 6 forward passes.


## Run 122
Value head has a 192-unit hidden layer and a 17-class linear output. Exact posterior inputs and exported full recurrent slots are available. Inspecting existing head reconstruction and recurrent tensor layout to build a factual-constrained, gradient-free dose-sensitivity edit. No edits. Counts: 122 Python runs, 6 forward passes.


## Run 123
Run 122 output was truncated by disassembly; no new model validation or edit resulted. Using bounded diagnostics for configuration, recurrent weights, and cached head arrays. No edits. Counts: 123 Python runs, 6 forward passes.


## Run 124
Constructed a gradient-free CPU one-step recurrent replay and tested activation, memory inclusion, and age-key conventions against original exported states. Best tests: [(7.2469563484191895, 0.14198486506938934, 'silu', True, 'before_kv', 1.0), (7.803891658782959, 0.1437201201915741, 'gelu', True, 'before_kv', 1.0), (12.443230628967285, 0.23995333909988403, 'silu', False, 'before_kv', 1.0), (13.114036560058594, 0.24768725037574768, 'gelu', False, 'before_kv', 1.0)]. No weights edited. Counts: 124 Python runs, 6 forward passes.


## Run 125
Initial CPU replay was not validated (best h RMSE 7.25, out RMSE 0.142). Tested key-only age projections, attention layouts, and independent FFN/age activations. Best refined tests: [(6.965161323547363, 0.13313856720924377, 'split', 'before_kv', 'gelu', 'silu', 'silu', 'gelu'), (6.990720272064209, 0.13382229208946228, 'split', 'before_kv', 'silu', 'silu', 'silu', 'gelu'), (6.9959635734558105, 0.13433274626731873, 'split', 'before_kv', 'gelu', 'silu', 'gelu', 'gelu')]. No weights edited. Counts: 125 Python runs, 6 forward passes.


## Run 126
Independent activation/layout tests did not validate replay. Tested memory observation filtering, padding, current-bundle inclusion, memory normalization, and age units. Best tests: [(6.930703639984131, 0.13412946462631226, True, False, False, 1.0, 1.0, 'before'), (6.954826831817627, 0.1345936805009842, True, True, False, 1.0, 1.0, 'before'), (6.965161323547363, 0.13313856720924377, True, False, False, 1.0, 1.0, 'none')]. No weights edited. Counts: 126 Python runs, 6 forward passes.


## Run 127
Memory conventions did not validate replay. Added the previously omitted ReLU option for all feed-forward/age layers. Best tests: [(6.965161323547363, 0.13313856720924377, 'gelu', 'silu', 'silu', 'gelu', 'before_kv'), (6.970487117767334, 0.1373649388551712, 'gelu', 'silu', 'relu', 'gelu', 'before_kv'), (6.970667839050293, 0.1330784112215042, 'relu', 'silu', 'silu', 'gelu', 'before_kv')]. No edits. Counts: 127 Python runs, 6 forward passes.


## Run 128
Activation substitutions did not validate recurrent replay. Re-reading the documented architecture and inspecting compact helper signatures to find a structural mismatch rather than continuing activation searches. No weights edited. Counts: 128 Python runs, 6 forward passes.


## Run 129
Architecture reread; prior output was truncated before helper internals. Inspecting replay implementation and complete core tensor layout directly. No weights edited. Counts: 129 Python runs, 6 forward passes.


## Run 130
Checking attention layout, model configuration, and alignment of replay inputs with exact encodings and exported posteriors. No weights edited. Counts: 130 Python runs, 6 forward passes.


## Run 131
Replay encodings/posteriors are exact and attention head count is correct. Coda inversion RMSE=2.37331e-05; isolated recurrent update via inverse output projection and fitted structural coefficients. No weights edited. Counts: 131 Python runs, 6 forward passes.


## Run 132
Coda inversion is stable, but no simple recurrent coefficient repair explains replay discrepancy. Will use exported states for exact factual constraints, and regard replay-based sensitivities as approximate until GPU validation. Recovering record evidence and factual head helpers. No weights edited. Counts: 132 Python runs, 6 forward passes.


## Run 133
Recovered early record evidence and tested independent self/cross attention layouts. Best candidates: [(6.965161323547363, 0.13313856720924377, 'split', 'split', 'before_kv', 'gelu', 'silu', 'silu'), (6.990720272064209, 0.13382230699062347, 'split', 'split', 'before_kv', 'silu', 'silu', 'silu'), (6.9959635734558105, 0.13433274626731873, 'split', 'split', 'before_kv', 'gelu', 'silu', 'gelu')]. No weights edited. Counts: 133 Python runs, 6 forward passes.


## Run 134
Joint attention layouts did not validate replay. Testing innovation decoding separately by measurement channel from exact original head inputs and hidden features, with stay-held-out validation. No edits. Counts: 134 Python runs, 6 forward passes.

## Run 135
Restart recovery: reading architecture, original record evidence, and live helper inventory. Forward 7 supplies 99 complete bundle states for two training stays; it is not itself replay validation. No weights edited. Counts: 135 Python runs, 7 forward passes.

## Run 136
Architecture recovered; previous output truncated before live inventory. Requesting bounded helper and array inventory. No edits. Counts: 136 Python runs, 7 forward passes.

## Run 137
Recovered helper names. Inspecting core tensor layout and exact head cache before choosing a factual-constrained intervention pathway. No edits. Counts: 137 Python runs, 7 forward passes.

## Run 138
Held-out innovation decoding is weak, particularly through value-head hidden features and alternate channels. Not treating this as a validated head-only repair. Inspecting replay implementation and dynamic causal estimates. No edits. Counts: 138 Python runs, 7 forwards.

## Runs 139–140
Run 139 failed before editing because mlp activation dispatch expected a callable. Run 140 repaired string/callable dispatch and checked head reconstruction: mean RMSE=1.7159, mean NLL difference=0.00029635. If embed.numeric.weight is edited, this is ONLY the temporary 1.5x centered dose/rate sensitivity probe; restore after states export. Counts: 140 Python runs, 7 forwards.

## Run 141
Forward 8 exported the temporary centered 1.5x numeric-dose probe states. Restored embed.numeric.weight to original; recovering architecture, causal evidence, and caches for an exact exported-state sensitivity comparison. Counts: 141 Python runs, 8 forward passes.

## Run 142
Numeric probe restoration confirmed. Architecture shows dose embeddings influence posterior and recurrent state, not only values. Recovering bounded cache inventory after run 141 output truncation. Counts: 142 Python runs, 8 forwards.

## Run 143
Focused head-helper and cache inspection; no weights edited. Counts: 143 Python runs, 8 forward passes.

## Runs 144–145
Run 144 stopped on state-row alignment before any comparison. Run 145 matches by stay and bundle time and reconstructs heads only on complete matches. No edits. Counts: 145 Python runs, 8 forwards.

## Run 146
Combined original exports 2 and 4 to align all 100 numeric-probe stays. Reconstructed paired value-head features; no recurrent replay required and no weights edited. Counts: 146 Python runs, 8 forwards.

## Run 147
Paired numeric probe shifts: glucose_finger mean -0.0908, MAP arterial +0.00016; factual head NLL changes tiny. Testing head-only sensitivity/factual-variance feasibility and checking record-effect units. No edits. Counts: 147 Python runs, 8 forwards.

## Run 148
Restart recovery. Probe/factual covariance bounds argue against blindly amplifying the value head. Reading architecture and bounded live-cache inventory before constructing a compensated, reversible dose-path edit. No edits. Counts: 148 Python runs, 8 forward passes.

## Run 149
Architecture confirms action embeddings affect posterior and recurrent state; no validated localized pathway yet. Recovering compact cache inventory and head construction to avoid truncated diagnostics. No edits. Counts: 149 Python runs, 8 forward passes.

## Run 150
Run 149 inventory exceeded output limit; no new validation or edit. Restricting output to head caches, action/latent tensors, and bounded helper internals. Counts: 150 Python runs, 8 forward passes.

## Run 151
Focused baseline-rollout and latent/action tensor recovery; broad inventories have been unhelpful because of output truncation. No edits. Counts: 151 Python runs, 8 forward passes.

## Run 152
Original rollout artifacts confirm under-response, especially NE x2. Investigating a posterior dose-minus-pre-dose-context pathway, instead of unconstrained output amplification. No edits. Counts: 152 Python runs, 8 forward passes.

## Run 153
Testing exact posterior reconstruction and stay-held-out current-dose/policy decoders from action and pre-dose context. No edits. Counts: 153 Python runs, 8 forward passes.

## Runs 155–156
Run 155 failed on Tensor.copy before changing any weights. Run 156 uses tensor-safe cloning, preserves the outstanding numeric embedding in recovery_numeric_154, and restores the original. Recovering architecture and bounded live inventory. Counts: 156 Python runs, 8 forward passes.

## Run 157
Original weights restored. Recovering record-derived causal estimates, posterior decoder validation, baseline rollouts, and exact editable pathways before choosing a candidate. Counts: 157 Python runs, 8 forward passes.

## Run 158
Record residual estimates consistently suggest insulin lowers glucose and NE raises MAP; one-step estimates alone are not rollout targets. Recovering lag evidence, baseline summaries, and the preserved numeric probe before GPU testing. Counts: 158 Python runs, 8 forward passes.

## Run 159
Preserved run-154 edit was exactly 5x original numeric columns 2 and 4, not the earlier 1.5x centered probe. Installed this simple 5x scaling as a reversible GPU experiment. Baseline enriched tuning shifts: insulin x2 -1.7733, half +0.8161; NE x2 +0.0479, half -0.4813. Estimated six-hour impulse responses imply larger accumulated effects. Counts: 159 Python runs, 8 forward passes.

## Forward 9–10 / Run 160
5x numeric dose/rate scaling: insulin x2 -2.449, half +2.651; NE x2 +0.043, half -0.323. Tuning factual NLL 1.53347 vs original 1.53139, delta +0.00208 (within bound on this batch). Keeping this as a reversible fallback while obtaining matched states and six-lag record-derived targets. Counts: 160 Python runs, 10 forward passes.

## Run 161
Forward 11 exported states under the provisional 5x numeric dose/rate edit. Recovering architecture and compact live inventory before matched-state analysis. The edit remains provisional; factual batch delta was +0.00208. Counts: 161 Python runs, 11 forward passes.

## Run 162
Architecture recovered: doses enter action encoder, posterior, prelude, and recurrent state; all heads share out. Recovering early causal evidence and exact-state caches to avoid another unvalidated replay. Counts: 162 Python runs, 11 forwards.

## Run 163
Early cross-fitted pre-dose residual slopes: insulin -2.21 mg/dL per unit/hour; NE +3.27 mmHg per mcg/kg/min, with alternate channels agreeing in sign. These are observational estimates, not ground truth or rollout targets. Recovering lag kernels and exact-state head helper. Counts: 163 Python runs, 11 forwards.

## Run 164
Six-lag kernels recovered exactly: insulin [-1.850,-2.256,-1.943,-1.561,-1.022,-0.690]; NE [3.351,2.621,1.717,0.943,0.494,0.409]. Constructed matched original/5x heads using exported states, without recurrent replay. Counts: 164 Python runs, 11 forwards.

## Run 165
5x matched-state changes: out RMS .03085, slots 1.66381, posterior .07330; approximate head NLL delta +.00309 finger and +.00051 arterial. Large slot changes are mostly not useful output changes. Testing posterior reconstruction and dose-minus-policy decoding for a localized correction. Counts: 165 Python runs, 11 forwards.

## Run 166
Posterior reconstruction from ds_in failed (best tested initially RMSE .218), so no posterior edit yet. Checking cache alignment, input/output layout, and held-out dose decoders before trusting this pathway. Counts: 166 Python runs, 11 forwards.

## Run 167
Aligned caches confirmed. Held-out active-dose RMSE: insulin .280 from action and .575 from context; NE .217 from action and .269 from context. Rebuilding posterior context from exported previous states to resolve remaining reconstruction error. Counts: 167 Python runs, 11 forwards.

## Run 168
Fresh context reduced posterior RMSE to .0186; examining earlier corrected rr_input4 cache. Estimated stay-cross-fitted latent observation directions per physical glucose/MAP unit, conditional on pre-bundle context. These directions are candidate pathways, not proof that injecting them will reproduce physiology. Counts: 168 Python runs, 11 forwards.

## Run 169
Restart recovery focused on exact architecture and causal-path caches. Fivefold numeric scaling remains provisional; posterior reconstruction is not yet exact, so no posterior edit is justified.

## Run 170
Architecture supports an action/posterior-to-state edit. Recovering only posterior inputs, physical observation directions, and early causal conclusions; no further weight changes.

## Run 171
Preparing a low-rank posterior dose-minus-policy correction. Checking final-layer activation/layout and available physical directions before any weight edit.

## Run 172
Using r168_in/r168_hidden rather than stale ds_in. Plan: fit dose-contrast decoders with explicit factual-zero penalties, then inject record-supported physical latent directions through z_post.2 only. Candidate remains unvalidated.

## Run 173
Compared preserved fresh contexts and encoder conventions. Dose contrasts will be anchored to the best factual input, so raw-encoder mismatch cannot create a spurious baseline correction. No new edit yet.

## Run 174
Built anchored counterfactual posterior features for x0, x0.5, x1.5, x2, x3, and fit closed-form two-output dose-contrast regressions with factual-zero penalties. Holdout is by stay. No new model weights installed.

## Run 175
Final-layer contrast decoder failed NE x2 despite reasonable pooled slopes. Tested input-level linear decoders conditional on active drug, with separate presence gates; this distinguishes genuine dose sensitivity from pooled-bin artifacts. No edit installed.

## Run 176
Installed reversible gated dose-residual posterior branch, gains I=4 and NE=4. Four low-impact GELU neurons repurposed; original removed contributions compensated by a held-out-checked regression. Numeric 5x fallback preserved in rp_fallback_numeric and restored to original in the active candidate. Edits: z_post.0/.2 weights and biases. Awaiting GPU factual and rollout validation.

## Forward 12–13 / Run 177
Posterior four-neuron branch gains 4/4: tuning factual NLL unchanged at reported precision (1.53139). Paired rollout shifts original -> candidate: insulin x2 -1.773 -> -1.811; half .816 -> .807; NE x2 .048 -> .070; half -.481 -> -.473. Branch is too weak; recovering exact construction and transmission pathway before a controlled gain or pathway test. Counts: 177 Python runs, 13 forwards.

## Run 178
Inspecting posterior gates and counterfactual activation before gain changes; recovering record-derived targets and data caches. No weights changed. Counts: 178 Python runs, 13 forwards.

## Forward 14 / Run 179
Exported paired states for posterior branch gain 4/4; comparing actual latent, slot and output changes to distinguish gate inactivity from weak transmission. No new edit. Counts: 179 Python runs, 14 forwards.

## Run 180
Actual gain-4 branch changes posterior RMS .07939 but out RMS only 6.75e-6 (drug-bundle out about 6–7e-7): active posterior, weak transmission. Installed reversible gain-200/200 probe with original numeric embedding. Six-lag observational x2 projections on enriched tuning reached hours: {'insulin': -11.309633077887218, 'norepinephrine': 4.568496535822766}. These are inferred projections, not supplied truth. Counts: 180 Python runs, 14 forwards.

## Run 181
Gain-200 posterior probe still yields negligible rollout changes (forward 16). This rejects gain amplification of the present posterior pathway as a useful repair. Recovering direct action/state architecture and factual forward 15 before replacing it. Counts: 181 Python runs, 16 forwards.

## Run 182
Forward 15 confirms posterior gain-200 factual NLL delta +0.000000663 on 5,128 readings, but forward 16 confirms ineffective rollouts. Recovering prior experiments and CPU helpers to design a direct state/readout correction.

## Run 183
Preserved gain-200 posterior tensors in saved_posterior183, then restored original z_post weights because the route is ineffective. All weights are now original. Bounded recovery of helpers and state caches.

## Run 184
Investigating direct value-head correction using existing dose-sensitive state contrasts, with factual-zero anchoring and stay-held-out validation. Recovering only necessary CPU helpers and prior approach summaries. No edits.

## Run 185
Measuring original/edited state transmission and direct readout cache availability before choosing an algebraic pathway edit. No edits.

## Run 186
Posterior-to-h gain is weak; full CPU core replay error remains too large. Checking whether a localized two-coordinate dose-residual carrier can drive z_in directly, instead of injecting into observational latent directions. No edits.

## Runs 187–188
Run 187 inspected 1.5x (not 5x) cached head contrasts; largest contrast/factual generalized sensitivity was 0.05786. Run 187 stopped on list/dict mismatch before any edits. Run 188 checks normalization and reconstructs matched 5x head features. No edits.

## Run 189
Matched original/5x states and value-head inputs are aligned. Recovering exact architecture, usable helpers, and early identification evidence before choosing a direct, factual-preserving dose pathway. No weights changed.

## Runs 190–191
Recovered pre-dose-history-adjusted associations: insulin negative, norepinephrine positive, with alternate channels broadly agreeing. Run 190 weight inspection failed because load_original returns torch tensors; run 191 converts explicitly to numpy. No weights changed.

## Run 192
Located exact latent tensor names and recovered posterior branch arrays and matched value-head cache layouts. No edits; testing whether a latent carrier can be connected rather than merely amplified.

## Run 193
z_in has substantial weights; weak transmission cannot be blamed on a zero projection. Fitted closed-form value-head tilt candidates against original/5x matched-state contrasts, with factual-zero penalties and stay-held-out checks. Targets use recovered six-lag record-derived kernels and centered numeric dose differences. No edit installed.

## Run 194
Direct output-head contrast fit rejected: factual-safe lambda=30 captured only ~2% of contrast targets; stronger fits violated the NLL bound. Recovering gated dose residuals for a separate latent carrier and direct state transmission. No edits installed.

## Run 195
Measured gated branch factual and held-out multiplier responses; selected two predictable posterior coordinates and fitted a closed-form z_in compensation from the remaining coordinates. No edits installed; compensation and carrier behavior must be validated before GPU testing.

## Run 196
Two latent coordinates [16,9] admit accurate factual compensation. Original insulin branch leaked on inactive bundles; tested 10x presence-gate margin. Derived glucose/MAP token directions from original observation encodings, using only supplied weights and vocab. No edit installed.

## Run 197
Strong presence gates eliminate inactive-bundle leakage. Recovering exact state equations and latent compensation before installing a direct carrier; observation directions alone are not proof of causal transmission. Counts: 197 Python runs, 16 forwards.

## Run 198
Architecture confirms the latent token enters prelude attention, but prior probes show negligible effective transmission. Inspecting carrier compensation and recovering numeric-dose fallback validation before choosing a candidate. Counts: 198 Python runs, 16 forwards.

## Run 199
Recovering carrier cache and actual saved forward summaries. Six-lag record-derived response kernels suggest substantial insulin and NE effects; large hidden-slot changes alone are not sufficient validation. No edits installed. Counts: 199 Python runs, 16 forwards.

## Run 200
Installed reversible two-coordinate latent carrier [16,9], four strongly gated GELU neurons, compensated removed posterior contributions and original z_in coordinates. Direct token directions: insulin -12 times glucose encoding derivative, NE +6 times MAP derivative. Numeric embedding remains original. Candidate requires paired GPU factual/rollout validation; no efficacy claimed. Counts: 200 Python runs, 16 forwards.

## Forward 17–18 / Run 201
Carrier gains 12/6: factual NLL unchanged at displayed precision; rollout insulin x2 -1.814 vs -1.773, half .791 vs .816; NE x2 .122 vs .048, half -.311 vs -.481. Rejected as ineffective; preserved tensors in carrier201_saved and restored all originals. Investigating attention/state transmission rather than further latent gain amplification. Counts: 201 Python runs, 18 forwards.

## Run 202
Testing the specific hypothesis that prelude attention ignores the latent token. Recovering CPU attention implementation and core scales to construct a dedicated carrier relay with algebraic factual compensation. All weights remain original. Counts: 202 Python runs, 18 forwards.

## Run 203
Recovered architecture and prior numeric-dose/identification summaries to choose a direct, testable transmission pathway. All weights remain original. Counts: 203 Python runs, 18 forwards.

## Run 204
Record diagnostics support opposing drug effects, conditional on pre-dose adjustment. Recovering numeric-edit evidence and exact attention helpers before designing a relay. No weights changed. Counts: 204 Python runs, 18 forwards.

## Run 205
Inspecting exact matched bundle inputs and prelude attention allocation to locate the failed latent transmission. Numeric 5x remains only a saved fallback, not an active edit. Counts: 205 Python runs, 18 forwards.

## Run 206
Prelude attention overwhelmingly reads action tokens and usually ignores latent tokens. Tested closed-form factual compensation for a uniform single-head carrier relay, with two reserved carrier features and stay-held-out local validation. No weights installed. Counts: 206 Python runs, 18 forwards.

## Run 207
Installed compensated uniform prelude attention head 2, with 46 factual features and two unused carrier features. Held-out local attention RMSE .03996 versus original RMS 11.655. No drug carrier installed yet; GPU factual validation is required. Counts: 207 Python runs, 18 forwards.

## Forward 19 / Run 208
Compensated uniform head baseline factual NLL delta 0.00054770 on 5128 readings, within .005 bound on this batch. Reconstructing saved dose carrier and exact counterfactual inputs before connecting it. Counts: 208 Python runs, 19 forwards.

## Run 209
Relay baseline passed the tested factual bound (+0.00054770 nats/reading). Recovering exact attention and carrier definitions before connecting them; no additional weights changed. Counts: 209 Python runs, 19 forwards.

## Run 210
Architecture confirms prelude attention is the carrier's only state entry. Inspecting relay reserved rows and factual carrier leakage before any new edit. Counts: 210 Python runs, 19 forwards.

## Run 211
Installed saved strong-gate latent carrier (insulin gain 12, NE gain 6) on top of compensated uniform attention relay. Verifying carrier residuals from installed weights rather than older ungated caches. This is a reversible transmission probe, not yet a validated repair. Counts: 211 Python runs, 19 forwards.

## Forward 20 / Run 212
On 25 enriched tuning stays, combined relay+carrier x2 insulin -1.925 versus original -2.062; half .953 versus .983; NE x2 .227 versus .113; half -.573 versus -.658. Natural relay output directions remain ineffective. Restored relay-only baseline; investigating dedicated carrier output channels and recovering earlier numeric-edit evidence. Counts: 212 Python runs, 20 forwards.

## Run 213
Testing whether low-variance pooled-token directions can carry dedicated dose signals, and recovering local state-update helpers for finite-difference direction selection. No new weights changed. Counts: 213 Python runs, 20 forwards.

## Runs 214–215
Local CPU core reconstruction is approximate (out RMSE .24–.25); run 214 stopped on an inventory-formatting error without editing weights. Run 215 constructs two low-background marker channels and validates dose responses on held-out stays. Any CPU-proposed output direction requires paired GPU verification. Counts: 215 Python runs, 20 forwards.


## Run 216
Restart recovery: marker dose contrasts are promising, but factual carrier leakage and NE x3 instability remain unresolved. Reading architecture, identification evidence, and saved implementation objects before selecting a transmission edit. No weights changed. Counts: 216 Python runs, 20 forwards.


## Run 217
Architecture recovery complete; prior output truncated before implementation inventory. Retrieving compact identification, numeric-amplification results, and latest marker/relay objects. No weights changed. Counts: 217 Python runs, 20 forwards.


## Runs 218–219
Run 218 inspection stopped on a numpy-module shape attribute; no edits changed. Run 219 uses narrow, type-checked inspection to recover transition outputs, marker scales, and head helpers. Counts: 219 Python runs, 20 forwards.


## Run 220
Measured symmetric finite-difference Jacobians from prelude attention output to h, out, and world view on 48 cached bundles. No gradients or optimizer used; no weights changed. Recovering value-head interface to select sign-correct marker output directions. Counts: 220 Python runs, 20 forwards.


## Run 221
Attention-to-output Jacobian is small. Fitted two closed-form finite-difference response directions against glucose and MAP heads, with alternate-channel and held-out local checks. Testing finite amplitudes before installation. No weights changed. Counts: 221 Python runs, 20 forwards.

## Run 223
Run 222 failed at an assertion before a new candidate was installed. Recovering architecture, record-derived effect evidence, and recent calculation objects. Current edits remain the compensated relay-only baseline; no efficacy inferred from the failed run.

## Run 224
Run 223 output was truncated before the candidate inventory. Narrow recovery of recent arrays/functions and earlier numeric-edit validation; no weights changed. Early adjusted associations are not being substituted for dynamic rollout targets.

## Run 225
Recovered 5x numeric fallback: factual +.00208 on one batch but weak/asymmetric rollout response. Inspecting marker/relay dimensions and finite-response helper before attempting a dedicated output route. No weights changed.

## Run 226
Checking finite-amplitude reading responses of the two closed-form attention directions and factual marker leakage. Large direction norms make infinitesimal Jacobian success insufficient. No weights changed.

## Run 227
Finite B221 responses are nonlinear at unit amplitudes and have cross-channel spillover. Testing closed-form factual marker compensation and lower-norm mean-response directions. No weights changed; preserved counterfactual contrast after compensation is required before installation.

## Runs 228–229
Run 228 stopped before installation: missing training-row mask in ridge RHS, and nt215 was not a verified factual token cache. Run 229 rebuilt posterior/latent tokens directly from candidate weights, corrected the mask, refit factual compensation, and installed the reversible two-marker candidate for paired GPU validation.

## Run 230
Forward 21 rejected: factual delta +0.19319 exceeds .005. Preserved candidate in rejected230 and restored all originals. Recovering architecture and cached identification before selecting a new route. Counts: 230 Python runs, 21 forwards.

## Run 231
Recovering record-derived dynamic targets and decomposition of rejected carrier. All weights remain original. Counts: 231 Python runs, 21 forwards.

## Run 232
Installed reversible marker-output shrinkage t=.10 from rejected229 toward validated relay baseline, retaining marker inputs. Large output norms may amplify small factual leakage; GPU factual and rollout tests will determine whether shrinking is viable. Counts: 232 Python runs, 21 forwards.

## Run 233
Forward22 t=.1 failed factual bound (+.12212). Rejected scaling hypothesis; restored tested relay-only baseline. Comparing latent-input changes and investigating a more direct recurrent injection. Counts: 233 Python runs, 22 forwards.

## Run 234
Installed earlier carrier211+relay without dedicated markers for an isolating factual test. Marker z_in replacement differs substantially from this candidate. Counts: 234 Python runs, 22 forwards.

## Run 235
Forward23 carrier211+relay factual delta -.00022 on 2879 readings. Failure localized to dedicated marker construction. Plan: retain validated latent mapping and derive marker projections from its dose contrasts without replacing z_in. Counts: 235 Python runs, 23 forwards.


## Run 236
Preserved the factual-safe carrier211+relay weights in safe236. Recovering architecture, record-derived effect evidence, and carrier construction before deriving markers without replacing z_in columns. The large column differences are a suspected failure source, not an isolated causal test. No weights changed. Counts: 236 Python runs, 23 forwards.


## Run 237
Run236 output was truncated before implementation details. Retrieving compact dynamic-effect evidence and safe posterior/cache definitions. Early adjusted associations support signs but are not being treated as multi-hour causal targets. No weights changed. Counts: 237 Python runs, 23 forwards.


## Run 238
Reconstructed safe/original latent tokens and normalized pooled inputs without replacing z_in. Retrieved compact counterfactual-cache and dynamic-estimate inventories. No weights changed. Counts: 238 Python runs, 23 forwards.


## Run 239
Built all ten safe-carrier dose contrasts. Tested linear projection separation, held-out factual leakage, and x0/.5/1.5/2/3 scaling. No weights changed. Counts: 239 Python runs, 23 forwards.


## Run 240
Rejected pooled markers: stopping/halving frequently had wrong-sign or cross-drug responses. Tested a closed-form constant-query/token-type key selector and exact latent-coordinate value projections, retaining the safe z_in mapping. No weights changed. Counts: 240 Python runs, 23 forwards.

## Run 241
Recovered architecture and identification notes before using the selective latent-token marker. Its monotonic dose contrasts are promising, but factual RMS is not evidence of a safe output edit. No weights changed. Counts: 241 Python runs, 23 forwards.

## Run 242
Architecture recovery confirms indirect attention-to-state transmission. Run241 output truncated before recent definitions; retrieved narrow inventory and numeric identification notes. No weights changed. Counts: 242 Python runs, 23 forwards.

## Run 243
Retrieved later dynamic-identification evidence and compensated-relay layout. Safe236 remains unchanged; selective relay will retain its z_in mapping. Counts: 243 Python runs, 23 forwards.

## Run 244
Reconstructed prelude attention features against cached originals and prepared selective-head compensation features while preserving safe236 latent weights. No weights changed. Counts: 244 Python runs, 23 forwards.

## Run 245
Attention reconstruction verified to <4e-6 RMSE. Fitted selective-head factual compensation and dose-invariant marker predictors with stay-held-out leakage/contrast checks. No weights changed. Counts: 245 Python runs, 23 forwards.

## Run 246
Perfect factual marker cancellation also eliminates dose contrasts. Evaluated intermediate invariance penalties and finite low-norm output responses; factual-versus-counterfactual tradeoff remains explicit. No weights changed. Counts: 246 Python runs, 23 forwards.

## Run 247
Installed reversible selective latent-attention candidate without replacing safe236 z_in: dose-invariant compensation weight1; insulin output -.5*Bmean, NE +.4*B221; rank48 attention factorization. Local finite responses only guide signs, not rollout efficacy. Paired GPU factual validation next. Counts: 247 Python runs, 23 forwards.

## Run 248
Selective candidate has intended local signs but held attention perturbation RMS4.07; this is not proof of factual safety. Selected random100 tuning stays for paired factual test and inspected causal-exposure units. No additional weights changed. Counts: 248 Python runs, 23 forwards.

## Run 249
Forward24 selective candidate failed factual fit: 1.64565 versus 1.61876, delta +.02689 on 4741 readings. Preserved as rejected249 and restored safe236. Recovering architecture and record-based identification before further edits. Counts: 249 Python runs, 24 forwards.

## Run 250
Run249 restored safe236; output truncated after architecture and early conditional-effect estimates. Narrow recovery of robustness, dynamic targets and candidate construction. No weights changed. Counts: 250 Python runs, 24 forwards.

## Runs 251–252
Forward25 safe236 rollout was essentially unchanged: insulin x2 -2.051, half .981, NE x2 .071, half -.396. Run251 verified head layout but stopped before edits on numpy.clone. Run252 corrected array conversion and installed quarter-strength selective marker output with retained factual compensation. Discrete-hour six-lag convolution targets: {'insulin_x2': -12.763239357347505, 'insulin_half': 6.3816196786737525, 'ne_x2': 4.142167540798611, 'ne_half': -2.0710837703993055}; conditional estimates, not known simulator truth. Counts: 252 Python runs, 25 forwards.

## Run 253
Forward27 quarter-strength selective marker did not improve the desired rollout effects (insulin x2 -2.003, half .818; NE x2 -.074, half -.154). Recovering architecture, factual26, and existing identification objects before choosing a different weight route. No weights changed. Counts: 253 Python runs, 27 forwards.

## Run 254
Recovered factual26: current candidate delta +.00247 nats, within bound, but forward27 shows no useful causal-response gain. Inspecting cached update functions and earlier identification evidence to locate the lost dose signal. No weights changed. Counts: 254 Python runs, 27 forwards.

## Run 255
Preserved quarter-strength candidate255 and restored safe236 because forward27 showed no useful effect gain. Recovering state/head functions and earlier unsuccessful head routes to distinguish lost state signal from weak decoding. Counts: 255 Python runs, 27 forwards.

## Run 256
Earlier value-head amplification failed its factual/contrast tradeoff; prelude-attention transmission was weak. Investigating core.b_in, the additive recurrent injection after the blocks, as a lower-distortion route. No weights changed. Counts: 256 Python runs, 27 forwards.

## Run 257
Measured finite-difference reading sensitivity to additive recurrent-state injection through core.b_in, using cached local CPU readout and no gradients. Computed minimum-norm channel-selective directions where the helper returned four readouts. No weights changed; local responses require GPU validation. Counts: 257 Python runs, 27 forwards.

## Run 258
Direct recurrent injection has selective, near-linear local reading effects: B257 RMS .241 glucose and .654 MAP per unit outcome. Reconstructing recorded and counterfactual prelude E features for a physically scaled, factual-anchored linear core.b_in edit. No weights changed. Counts: 258 Python runs, 27 forwards.

## Runs 259–260
CF token recovery succeeded: pooled features are mean-normalized tokens; all ten recovered action tokens satisfy LN mean/variance checks. Run259 then failed only in an inventory printer using torch.size as a number. Run260 fixes inventory and recovers physical-dose/carrier scaling. No weights changed. Counts: 260 Python runs, 27 forwards.

## Run 261
Run260 confirmed ds_rates physical-dose alignment but broad function printing truncated useful output. Recovering architecture and compact recurrent-input objects before a factual-anchored core.b_in edit. No weights changed. Counts: 261 Python runs, 27 forwards.

## Run 262
Architecture confirms core.b_in is an additive recurrent injection after the blocks. Cached factualE258 and all ten Es259 counterfactual features are available. Checking local response units, physical-rate alignment and record-effect evidence before fitting a factual-anchored rank-two edit. No weights changed. Counts: 262 Python runs, 27 forwards.

## Run 263
B257 produces approximately unit, channel-selective local reading shifts for both glucose or both MAP channels. Fitted rank-two E projections with factual-zero penalties and physical-rate counterfactual targets over x0/.5/1.5/2/3, checking stay-held-out leakage and response. No weights changed; local response does not establish rollout efficacy or factual safety. Counts: 263 Python runs, 27 forwards.

## Run 264
Installed reversible direct recurrent injection: original core.b_in plus B257 @ diag([-2,4]) @ fits263[1].T, retaining safe236 latent/relay weights. Projection held-out doubling ratios are .568 insulin/.727 NE; factual leakage requires paired GPU testing. Lower-dose and x3 response recovery is weaker, so this is an experimental candidate rather than a validated repair. Counts: 264 Python runs, 27 forwards.

## Run 265
Forward28 direct b_in candidate catastrophically failed factual fit: 5.8288 versus 1.53191 nats on 5059 readings. Removed core.b_in edit immediately, restoring safe236. Frozen-state leakage checks did not predict trajectory damage; investigating feedback/stability before any further direct injection. Counts: 265 Python runs, 28 forwards.

## Run 266
Failure diagnosis: P264.T @ (delta_in*B257) has entries up to 61; the edit reads its own injected directions and creates strong feedback. Fitted immediate-injection-orthogonal alternatives and inspected local coda/state implementation to assess transformed feedback. Safe236 remains installed. Counts: 266 Python runs, 28 forwards.

## Run 267
Recovered exact architecture and current injection objects after immediate-feedback orthogonalization. Safe236 remains installed; no weights changed. Counts: 267 Python runs, 28 forwards.

## Run 268
Recovered local transition/coda path, feature-cache alignment and earlier record-identification notes. No weights changed. Counts: 268 Python runs, 28 forwards.

## Run 269
Reconstructed prelude, recurrent injection and coda locally; measured transformed slot-feedback of existing dose projections. Recovered early identification and posterior-input alignment. Safe236 unchanged. Counts: 269 Python runs, 28 forwards.

## Run 270
Computed finite-difference transformed feedback including current posterior/latent-token response; fitted feedback-penalized recurrent injection directions without gradients. No weights changed. Counts: 270 Python runs, 28 forwards.

## Run 271
Installed candidate271: original core.b_in plus feedback-penalized B270[10000] @ diag([-1,2]) @ immediate-null P266[1].T; retained safe236 edits. Local feedback was substantially reduced with unit channel sensitivity retained, but reconstructed E mismatch means GPU factual validation is essential. Counts: 271 Python runs, 28 forwards.

## Run 272
Forward29 candidate271 failed factual fit: 2.99429 versus 1.53191, delta +1.46238. Saved rejected272 and removed core.b_in immediately, restoring safe236. Auditing mismatch between fitted prelude features and verified CPU reconstruction before another injection. Counts: 272 Python runs, 29 forwards.

## Run 273
Restart recovery: safe236 remains installed after both failed recurrent-injection candidates. Run272 exposed a major cached-versus-reconstructed slot-token mismatch. Reading architecture and recovering verified reconstruction and record-identification objects before any new edit. Counts: 273 Python runs, 29 forwards.

## Run 274
Architecture recovered; prior output truncation hid the needed reconstruction inventory. Compact recovery of verified recurrent features and dynamic record-effect evidence. No weights changed. Counts: 274 Python runs, 29 forwards.

## Run 276
Numerically auditing time decay, slot residuals, attention layout and current-versus-cached weights. Built fresh hflow/slot tokens from exported original states. No weights changed. Counts: 276 Python runs, 29 forwards.

## Runs 277–278
Corrected rounded-minute hflow matches verified replay exactly; old cached slot residuals differ by RMS23.74. Run277 stopped on a NumPy/Torch conversion after building finite corrected features. Completing posterior/prelude checks and recovering counterfactual action caches before fitting. No weights changed. Counts: 278 Python runs, 29 forwards.

## Run 279
Independent prelude matches replay exactly. Corrected E exposes old projection leakage RMS[46.51,9.36], so old dose projections are rejected. Rebuilt dose-edited action vectors for x0/.5/1.5/2/3, anchored to factual action encoding. Preparing GPU states for current safe236 on the same 100 training stays, to fit on actual edited-model trajectories. Counts: 279 Python runs, 29 forwards.

## Run 280
Forward30 exported 5215 actual safe236 bundle states on the aligned 100 training stays. Recovering architecture and reconstruction functions before fitting corrected factual-anchored features. No weights changed. Counts: 280 Python runs, 30 forwards.

## Run 281
Run280 output was truncated by verbose function defaults. Compact helper recovery and export alignment audit; rebuilt current incoming h for noninitial bundles using exported previous h and verified rounded-minute dt. No weights changed. Counts: 281 Python runs, 30 forwards.

## Run 282
Rebuilt current posterior from aligned exported previous states; posterior RMS/max error 1.11971/15.2204. Built 0 corrected counterfactual prelude feature sets if the reconstruction passed. No weights changed. Counts: 282 Python runs, 30 forwards.

## Run 283
Run282 posterior reconstruction failed (RMS1.12, max15.22); correctly rejected before fitting or editing. Auditing original-versus-current posterior, source-export flow alignment and verified local replay components to locate the error. Safe236 unchanged. Counts: 283 Python runs, 30 forwards.

## Run 284
Audited cached contexts and posterior activations. Best original: rp_ctx/gelu, RMS/max 0.163024/2.63526; best current normalization core.out_norm, RMS/max 1.0208/12.9035. Built 0 feature sets only if both checks passed. No weights changed. Counts: 284 Python runs, 30 forwards.

## Run 285
Posterior-from-context reconstruction remains unvalidated. Built actual factual E using GPU-exported posterior, and ten provisional action-only dose contrasts with that posterior held fixed; independently checked original factual tokens against verified replay. No weights changed. Counts: 285 Python runs, 30 forwards.

## Run 286
Exact factual prelude features confirmed via exported z and original replay. Fitted analytical rank-two dose projections across x0/.5/1.5/2/3 with factual penalties, immediate-injection null constraints and optional four-direction approximate transformed-feedback nulling; evaluated on 25 held-out stays. No weights changed. Posterior dose response remains a limitation requiring GPU testing. Counts: 286 Python runs, 30 forwards.

## Run 287
Restart recovery of architecture, record-effect identification, safe236 evidence and corrected projection objects. No weights changed. Counts: 287 Python runs, 30 forwards.

## Run 288
Run287 output truncation prevented recovery of current validation. Compact retrieval of safe236 evidence, record-effect conclusions, corrected projections and forward summaries. No weights changed. Counts: 288 Python runs, 30 forwards.

## Run 289
Installed reversible corrected-feature recurrent candidate: original b_in plus B286 @ diag([-2,4]) @ fits286[feedback4,10].T, retaining baseline latent/relay edits. Immediate injection feedback is constrained; longer-loop and posterior-dose feedback remain unvalidated. Requires paired factual gate before rollout testing. Counts: 289 Python runs, 30 forwards.

## Run 290
Forward31 corrected candidate failed: NLL 1.61612 versus 1.53191 (+.08421), concentrated in MAP. Rejected and replaced with gains [-3,.4] using the same corrected feedback-constrained projection, relative to original b_in. Requires new factual gate. Counts: 290 Python runs, 31 forwards.

## Run 291
Forward33 current [-3,.4] recurrent candidate predicts shifts [-4.013,+1.672,+.279,-.465], versus original [-1.731,+.798,+.048,-.481]. Recovering record-based effect estimates and Forward32 factual gate before further edits. No weights changed. Counts: 291 Python runs, 33 forwards.

## Run 292
Run291 output truncated before safety evidence. Saved current full edit checkpoint292 and performed compact recovery of Forward32 factual gate, projection objects, and record-based effect notes. No weights changed. Counts: 292 Python runs, 33 forwards.

## Run 293
Forward32 current candidate passes factual gate on 5059 readings: 1.53096444 edited versus 1.53191401 original, delta -.00094958. Recovering complete dynamic record-effect evidence and selected 100 independent tuning stays for validation. No weights changed. Counts: 293 Python runs, 33 forwards.

## Run 294
Forward34 current [-3,.4] candidate worsens independent factual NLL by +0.00596237, above .005 on this batch. Saved in checkpoint292; removed b_in edit to isolate safe236 baseline. Computed matched six-lag record-derived targets where rollout columns permit; these retain conditional-exogeneity and omitted-tail uncertainty. Counts: 294 Python runs, 34 forwards.

## Run 295
Forward35 safe236 baseline independent factual delta approximately +.00031; the recurrent injection caused most of Forward34 loss. Matched six-lag estimates on Forward33 reached rows: insulin x2 -11.20288, half +5.60144; NE x2 +4.56850, half -2.28425, conditional on identification assumptions. Installed feedback4 factual-penalty100 projection with gains [-6,2], retaining safe236. This is unvalidated and reversible. Counts: 295 Python runs, 35 forwards.

## Run 296
Forward36 candidate295 independent factual delta +.00155 passes the local gate. Forward37 matched rollout shifts [-2.482,+1.130,+.219,-.431], still much smaller than conditional six-lag record estimates. Saved candidate295 in kernel. Installed mixed candidate296: feedback4 penalty10 insulin column gain -4, penalty100 NE column gain 2, retaining safe236 edits. This is unvalidated; evaluate aggregate factual fit and rollout response. Counts: 296 Python runs, 37 forwards.

## Run 297
Restart recovery after Forward42: current rollout shifts [-4.754, 2.098, .329, -.395]. Recovering architecture, Forward38–41 factual gates, exact edit objects, and record-derived effect evidence. No weights changed. Counts: 297 Python runs, 42 forwards.

## Run 298
Compact audit of Forward38–41 and saved all current weights as current298. Current mixed candidate factual delta over 18699 readings: 0.00719430; stay-bootstrap 95% interval [0.0046131268144528345, 0.009652884790990345]. Recovering safer projection construction before final adjustment. No weights changed. Counts: 298 Python runs, 42 forwards.

## Run 299
Rejected candidate296: all-400 tuning factual delta +0.00719430, bootstrap interval approximately [0.00461,0.00965]. Installed candidate299 relative to original b_in: insulin direction -2*feedback4_penalty10[:,0]-3*feedback4_penalty100[:,0], NE direction 4*feedback4_penalty100[:,1], output directions B286. Retained safe236 latent/prelude edits. Saved candidate299 and fallback299 (same insulin, NE gain2). No gradients used. Record-derived six-lag matched targets remain approximately [-11.20,+5.60,+4.57,-2.28], conditional on sequential exogeneity and omitted-tail assumptions, not known ground truth. Candidate requires independent factual and rollout validation. Counts: 299 Python runs, 42 forwards.

## Run 300 — final installed weights, validation pending
Budget: 300/300 Python runs, 44/50 forward passes.
Candidate299 failed the difficult tuning batch: factual delta approximately
+0.00796. Its matched rollout shifts were [-3.597,+1.585,+0.257,-0.561].
Increasing the NE gain from 2 to 4 did not produce a compelling gain in effect
accuracy and increased MAP factual loss. Candidate296 had already failed the
full-400 tuning gate (+0.0071943), so neither is the submitted candidate.

Final installed correction, relative to ORIGINAL core.b_in.weight:
outer(B286[:,0], -1.6*fits286[feedback4,10][:,0]
                    -2.4*fits286[feedback4,100][:,0])
+ outer(B286[:,1], 2*fits286[feedback4,100][:,1]).
This is 80% of candidate299's insulin blend and restores the tested smaller
NE gain. Retained the nine safe236 prelude/posterior/latent-input edits.
All edits are analytical/statistical; no gradients or weight optimiser used.
The final complete checkpoint is retained as final300 in the kernel.

Evidence from training records: stay-separated adjustment of dose innovations
and six-hour deconvolution supported insulin-negative and NE-positive effects.
Recovered impulse kernels:
insulin [-1.850,-2.256,-1.943,-1.561,-1.022,-0.690];
NE [3.351,2.621,1.717,0.943,0.494,0.409].
On the matched 99-stay rollout cohort, conditional six-lag targets were
approximately [-11.20,+5.60,+4.57,-2.28]. These are observational estimates,
requiring sequential conditional exogeneity and adequate adjustment;
unobserved confounding and omitted response tails are not ruled out.
No medical knowledge was used as effect evidence.

The edit is a partial repair, not a demonstrated match to those targets.
Provisional expected final rollout shifts are roughly
[-3.2,+1.4,+0.25,-0.45], subject to rollout and cohort variation.
Remaining plan: four paired factual passes covering all 400 tuning stays,
then two paired dose rollouts. The hard factual criterion is <=+0.005 nats.
Stopping, x1.5/x3, hour6/hour16 starts and secondary-channel counterfactuals
are not directly validated by the available forward request. Projections
included local dose contrasts at x0/.5/1.5/2/3, but that is not a rollout
validation. All-output log-loss change has not been measured by these
value-NLL reports. Final validation outcomes must be reported honestly.
