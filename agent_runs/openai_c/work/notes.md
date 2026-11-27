# World-model repair notebook

## Initial inspection
- Python runs: 1; forward passes: 0.
- No weight edits made.
- Read architecture, record definitions, vocabulary and tensor shapes first.
- Drug effects are unknown. No medical assumptions will be used.
- Plan: establish effects from treatment variation in the records, inspect the model's corresponding predictions, then test targeted weight edits against both dose rollouts and the factual NLL constraint.

## Inspection, run 2
- First output was truncated after the vocabulary; loaded records and requested compact summaries.
- Architecture separates action encoding from observation encoding. Candidate edits must still be justified by estimated drug effects.
- No weight edits; no forward passes.

## Run 3: hourly panels and preliminary regressions
- Infusions begin at hour + 0.0833, after readings, and run one hour.
- Built complete -1..24 panels; absent infusion means rate zero.
- Preliminary adjusted slopes are associations, not yet established causal effects. Need nonlinear adjustment and independent checks.
- Sensor/dose summaries printed. Preliminary estimates: {('glucose_finger', 'insulin'): {'n': 400154, 'raw_slope': np.float64(27.415433254830994), 'adjusted_slope': np.float64(-0.813008629750258), 'cluster_se': np.float64(0.04208575715363863), 'residual_rmse': np.float64(13.420845287662623)}, ('glucose_lab', 'insulin'): {'n': 143791, 'raw_slope': np.float64(30.153346410333576), 'adjusted_slope': np.float64(-0.7489972769612019), 'cluster_se': np.float64(0.06732977710214152), 'residual_rmse': np.float64(11.56030767772446)}, ('map_arterial', 'norepinephrine'): {'n': 328940, 'raw_slope': np.float64(-30.026406261425024), 'adjusted_slope': np.float64(0.8529804747023586), 'cluster_se': np.float64(0.08565469619687506), 'residual_rmse': np.float64(4.43480783695137)}, ('map_cuff', 'norepinephrine'): {'n': 439865, 'raw_slope': np.float64(-33.90418077217516), 'adjusted_slope': np.float64(-4.774212872772716), 'cluster_se': np.float64(0.1866481398966458), 'residual_rmse': np.float64(7.695480574994642)}}
- Probe stays: tuning IDs 2020000000..2020000099; independent validation IDs 2020000200..2020000299.
- Python runs: 3; forward passes: 0; no edits.

## Run 4 / forwards 1–2: causal diagnostics and baselines
- Original probe factual NLL: 1.63461; allowed increase 0.005.
- Original dose rollouts: {'insulin_x2': -1.847, 'insulin_half': 0.777, 'ne_x2': -0.498, 'ne_half': -0.6}
- Insulin threshold estimates: {('glucose_finger', 1, 6): {'n': 94569, 'rate_jump': 0.676820167360905, 'outcome_jump': -1.469092110147066, 'effect': -2.1705796916120748, 'se': 0.2823686119400948}, ('glucose_finger', 1, 10): {'n': 158690, 'rate_jump': 0.800859016409116, 'outcome_jump': -1.7400345142239886, 'effect': -2.172710150690366, 'se': 0.18469317346762512}, ('glucose_finger', 1, 14): {'n': 223106, 'rate_jump': 0.8605678363604209, 'outcome_jump': -1.767739842270549, 'effect': -2.0541551375505844, 'se': 0.14557496818862847}, ('glucose_lab', 1, 6): {'n': 22161, 'rate_jump': 0.6106859197361182, 'outcome_jump': -0.9623250088225842, 'effect': -1.575810048540847, 'se': 0.4515120296985709}, ('glucose_lab', 1, 10): {'n': 36635, 'rate_jump': 0.7512887970002406, 'outcome_jump': -1.1779938611589744, 'effect': -1.5679640983101166, 'se': 0.2826879624651828}, ('glucose_lab', 1, 14): {'n': 50919, 'rate_jump': 0.8293928985338691, 'outcome_jump': -1.2996262711630073, 'effect': -1.566960934269364, 'se': 0.21622340829361023}, ('glucose_finger', 2, 10): {'n': 146117, 'rate_jump': 0.7829977388109858, 'outcome_jump': -1.5017770940980744, 'effect': -1.9179839476657807, 'se': 0.204116411045284}, ('glucose_finger', 4, 10): {'n': 130544, 'rate_jump': 0.7679516563693188, 'outcome_jump': -0.9530878642549095, 'effect': -1.2410779459228825, 'se': 0.24394129883778568}}
- Prior-reading placebo: {'n': 157736, 'rate_jump': 0.7725678382439083, 'outcome_jump': 0.06211945739767062, 'effect': 0.08040647606930126, 'se': 0.19329795224019525}
- Nonlinear cross-fitted estimates: {('glucose_finger', 'all'): {'n': 400154, 'effect': -1.9205123153267514, 'se': 0.046687521700809545, 'dose_residual_sd': 0.49018342634703577, 'outcome_residual_sd': 12.422922826191442}, ('glucose_finger', 'fresh'): {'n': 298450, 'effect': -1.9349220454366567, 'se': 0.047097440462058895, 'dose_residual_sd': 0.5639277025139702, 'outcome_residual_sd': 12.3318485715214}, ('map_arterial', 'all'): {'n': 328940, 'effect': 3.062084804446521, 'se': 0.08898857289398902, 'dose_residual_sd': 0.09352020066962713, 'outcome_residual_sd': 4.325493241423042}, ('map_arterial', 'fresh'): {'n': 312563, 'effect': 3.1166978915729437, 'se': 0.0891845742058681, 'dose_residual_sd': 0.09349417174331442, 'outcome_residual_sd': 4.295459295140099}}
- These analyses use only records and pre-infusion controls; cross-fitting separates stays. Assess identification assumptions and agreement before choosing target effects.
- Python runs: 4; forward passes: 2; no edits.

## Run 5 / forward 3: persistence and state capture
- Original states obtained on training IDs 1010000000..1010000099, all bundles (4035 rows; 992-dimensional state).
- Dynamic moments: {'insulin': [{'h': 1, 'n': 223106, 'effect': -2.054155137550587, 'se': 0.1455749681886286, 'phi': [1.0]}, {'h': 2, 'n': 205065, 'effect': -1.9215505007955112, 'se': 0.15973192071265485, 'phi': [1.0, 0.025215871026567586]}, {'h': 4, 'n': 182164, 'effect': -1.30868113770493, 'se': 0.19061378818234964, 'phi': [1.0, 0.04948621470637709, -0.005534859740399555, -0.01710917618348411]}, {'h': 6, 'n': 163583, 'effect': -0.6953659704972652, 'se': 0.2156277623132073, 'phi': [1.0, 0.05780741421219487, -0.005006001563280803, -0.020813196722217174, -0.03845105671049845, -0.02815311855012386]}], 'norepinephrine': [{'h': 1, 'n': 312563, 'effect': 3.116538954905723, 'se': 0.08932562245910886, 'phi': [1.0]}, {'h': 2, 'n': 299506, 'effect': 4.860383476324453, 'se': 0.1068007636923712, 'phi': [1.0, 0.7508293535077544]}, {'h': 4, 'n': 273503, 'effect': 5.158283324771074, 'se': 0.13259557350076712, 'phi': [1.0, 0.7560151330798912, 0.5911606243095916, 0.4716599701183076]}, {'h': 6, 'n': 247499, 'effect': 3.9048242277985303, 'se': 0.15550859589971106, 'phi': [1.0, 0.7613918338652426, 0.5983575838047945, 0.4797127918586145, 0.3880467773670123, 0.3179327899297267]}]}
- Persistence fit accounting for later treatment responses: {'insulin': {'beta': -2.1126230003348483, 'rho': 0.8412020161353004, 'observed': [-2.054155137550587, -1.9215505007955112, -1.30868113770493, -0.6953659704972652], 'fitted': [-2.1126230003348483, -1.8304143563196857, -1.2855404663702767, -0.7857956242711385], 'standardized_residuals': [-0.40163404128999375, 0.5705568684657099, 0.12140082601220833, -0.419378529015762]}, 'norepinephrine': {'beta': 3.2907376107066626, 'rho': 0.672711214384998, 'observed': [3.116538954905723, 4.860383476324453, 5.158283324771074, 3.9048242277985303], 'fitted': [3.2907376107066626, 4.6844984871314015, 4.988418327009077, 4.185545043195627], 'standardized_residuals': [1.950153281950914, -1.646851418587893, -1.2810759309473805, 1.8051787669546966]}}
- Approximate fixed-dose-edit targets, probe: {'insulin_x2': {'mean': -10.760784323326371, 'n_stay_hours': 643, 'n_stays': 60}, 'insulin_half': {'mean': 5.380392161663186, 'n_stay_hours': 643, 'n_stays': 60}, 'ne_x2': {'mean': 4.01609433089793, 'n_stay_hours': 474, 'n_stays': 42}, 'ne_half': {'mean': -2.008047165448965, 'n_stay_hours': 474, 'n_stays': 42}}
- Approximate targets, training population: {'insulin_x2': {'mean': -11.283462940451336, 'n_stay_hours': 175008, 'n_stays': 16457}, 'insulin_half': {'mean': 5.641731470225668, 'n_stay_hours': 175008, 'n_stays': 16457}, 'ne_x2': {'mean': 4.228313852084173, 'n_stay_hours': 127404, 'n_stays': 11318}, 'ne_half': {'mean': -2.1141569260420865, 'n_stay_hours': 127404, 'n_stays': 11318}}
- Targets remain model-based estimates from records, not observed counterfactual truth. Check horizon-fit residuals and method agreement.
- Python runs: 5; forward passes: 3; no edits.

## Run 6: corrected targets, state directions, candidate A
- Earlier rollout targets included unrecorded sensor-hours; superseded by the sensor-observed masks here.
- Corrected probe targets: {'insulin_x2': {'mean': -12.036254091096144, 'n_stay_hours': 554, 'n_stays': 60}, 'insulin_half': {'mean': 6.018127045548072, 'n_stay_hours': 554, 'n_stays': 60}, 'ne_x2': {'mean': 4.732347324472345, 'n_stay_hours': 281, 'n_stays': 26}, 'ne_half': {'mean': -2.3661736622361724, 'n_stay_hours': 281, 'n_stays': 26}}
- Corrected training-population targets: {'insulin_x2': {'mean': -12.51486026649639, 'n_stay_hours': 152158, 'n_stays': 16457}, 'insulin_half': {'mean': 6.257430133248195, 'n_stay_hours': 152158, 'n_stays': 16457}, 'ne_x2': {'mean': 4.6728108030283355, 'n_stay_hours': 79814, 'n_stays': 7206}, 'ne_half': {'mean': -2.3364054015141678, 'n_stay_hours': 79814, 'n_stays': 7206}}
- Estimated conditional dose-innovation directions from action-bundle state changes, controlling pre-action states and record histories. Directions saved in dose_directions; not applied yet.
- Candidate A: 30 rank-one output projections erase contrasts between nonzero dose/rate bin rows while preserving bin 0 and the common nonzero-bin component. Numeric inputs unchanged.
- Hypothesis: bin lookup artifacts cause irregular dose response. Require factual NLL increase <=0.005, then inspect dose rollouts.
- Python runs: 6; forward passes: 3.

## Run 7 / forwards 4–5: candidate A rejected
- Re-read architecture after restart.
- Candidate A factual NLL 1.63557 versus original 1.63461: +0.00096.
- Candidate A rollouts: insulin x2 -1.745, half +0.388; NE x2 -0.492, half +0.127.
- Bin-contrast erasure does not explain the causal failure; saved operations and cleared them.
- Next: inspect latent/action pathways and recovered state directions before constructing a targeted edit.
- Python runs: 8; forward passes: 5.

## Runs 9–10: candidate B
- Run 9 inspection confirmed z_post.0.weight is (192,576), then failed during an optional vocabulary print before any edit.
- Run 10 erased its action-input block: last 192 columns of [context, observation, action].
- Context/observation inputs, direct prelude action token, measurement heads and numeric embeddings unchanged.
- Hypothesis: action-dependent latent inference counteracts direct treatment dynamics. This requires experimental validation.
- Python runs: 10; forward passes: 5.

## Runs 13–14 / forward 8: state-to-reading surrogates
- Run 13 stopped on CSV column mismatch (observed versus value); run 14 corrected it. No weight edits.
- Original training factual NLL: 1.63841 on the 100 captured stays.
- Stay-held-out RMSE: {'glucose_finger': {0.1: 3.4756783472201076, 1.0: 4.0981795062462485, 10.0: 5.103901005035651, 100.0: 8.159300872225991}, 'map_arterial': {0.1: 1.3264895951311222, 1.0: 1.4700455248823527, 10.0: 1.7328266377334218, 100.0: 2.939310343816757}}
- Dose/readout diagnostics: {'glucose_finger': {'insulin': {'linear_response': -0.3193425773747281, 'raw': {'dose_projection': 0.035506415420687595, 'readout_projection': -8.993940210271546, 'rank1_response_per_alpha': -0.3193425773747279, 'factual_projection_mean': 0.8932123197462973, 'factual_projection_sd': 0.5319561660841933}, 'white': {'dose_projection': 0.02792455974508839, 'readout_projection': 5.411467559756522, 'rank1_response_per_alpha': 0.15111284918102866, 'factual_projection_mean': -0.0121715256400934, 'factual_projection_sd': 0.13200939142109847}}, 'norepinephrine': {'linear_response': -0.8611741607585301, 'raw': {'dose_projection': 0.11995085977817649, 'readout_projection': -7.17939131366868, 'rank1_response_per_alpha': -0.8611741607585301, 'factual_projection_mean': 0.36390079827860783, 'factual_projection_sd': 0.48013817674610426}, 'white': {'dose_projection': 0.09488102309376231, 'readout_projection': -0.09747847305508128, 'rank1_response_per_alpha': -0.009248857253083855, 'factual_projection_mean': -0.0059262774053348505, 'factual_projection_sd': 0.13120332501365656}}}, 'map_arterial': {'insulin': {'linear_response': 0.0975860107614033, 'raw': {'dose_projection': 0.035506415420687595, 'readout_projection': 2.7484050306172376, 'rank1_response_per_alpha': 0.09758601076140325, 'factual_projection_mean': 0.8601089190640421, 'factual_projection_sd': 0.4474455340771317}, 'white': {'dose_projection': 0.02792455974508839, 'readout_projection': 0.261679415173908, 'rank1_response_per_alpha': 0.007307282463083584, 'factual_projection_mean': -0.001379855473929868, 'factual_projection_sd': 0.13587645511568808}}, 'norepinephrine': {'linear_response': 0.8079921452484369, 'raw': {'dose_projection': 0.11995085977817649, 'readout_projection': 6.7360262922887415, 'rank1_response_per_alpha': 0.8079921452484369, 'factual_projection_mean': 0.5156701322550686, 'factual_projection_sd': 0.5009638916755382}, 'white': {'dose_projection': 0.09488102309376231, 'readout_projection': 1.0660110224997374, 'rank1_response_per_alpha': 0.10114421644400276, 'factual_projection_mean': 0.019418567291535963, 'factual_projection_sd': 0.12360463576393582}}}}
- These regressions approximate head sensitivity; forward tests remain decisive.
- Python runs: 14; forward passes: 8.

## Run 15: candidate C, action numeric amplification
- Head surrogates give immediate insulin sensitivity -0.319 per unit and NE sensitivity +0.808 per rate unit, but are approximate.
- Whitened directions overlap (cosine 0.848); whitening reverses the estimated insulin readout sign. Direct head amplification is not justified yet.
- Applied one input-side rank-one edit to embed.numeric.weight: multiply the equal z_dose/z_rate signal by 4. Measurement numeric inputs, presence flags, bins, and reading registers are unchanged.
- Test factual NLL first, then dose rollouts if the fit holds.
- Python runs: 15; forward passes: 8.

## Run 16 / forwards 9–10: candidate C results
- Numeric dose/rate gain 4: factual NLL 1.63584 versus 1.63461 (+0.00123).
- Probe rollouts: insulin x2 -2.503, half +1.691; NE x2 +0.127, half -0.003.
- Modest insulin improvement, no adequate NE response. Retained while investigating targeted directions.
- Python runs: 16; forward passes: 10.

## Run 18: numeric gain 16
- Latent-posterior action erasure had negligible factual and rollout effects (forwards 6–7).
- Replaced numeric gain 4 with gain 16; no other edits.
- Test whether stronger numeric signal remains factual-safe and avoids dose-response saturation.
- Python runs: 18; forward passes: 10.

## Forwards 11–12 / run 19: numeric saturation
- Gain 16 factual increase +0.00246, but rollouts insulin -2.231/+1.531 and NE +0.157/+0.085.
- Numeric gain saturates; restored gain 4 (-2.503/+1.691, NE +0.127/-0.003).
- Investigating dose-sensitive recurrent/latent directions with low factual second moment.
- Python runs: 19; forward passes: 12.

## Restart / Python run 20
- Forward 13 saved paired original/gain-4 states on 100 training stays.
- Re-inspecting architecture, earlier identification evidence, and persistent variables before designing a targeted recurrent edit.
- Python runs: 20; forward passes: 13.

## Python run 21: paired-state geometry
- Compared gain-4 and original states using generalized delta/factual second moments.
- Ratios: {'out': 1.1637545257572386, 'h0': 1.2496169933065107, 'h1': 1.2495189336363508, 'h2': 1.1449731622522863, 'h3': 1.2496636229526032, 'z': 0.43184577403443003}
- Python runs: 21; forward passes: 13.

## Python run 22: selective correction design
- Examined stored dose-state estimators and inferred the slot-to-out normalization from observable states, without accessing weight files.
- Slot-to-out reconstruction RMSE: 7.449458051528166e-08
- Python runs: 22; forward passes: 13; edits unchanged.

## Python run 23: calibrated slot shear design
- Designed separate minimum-variance insulin/NE sensors with zero factual mean, and reading-output directions orthogonal to the sensors.
- Candidate matrices add approximate immediate responses -1.5 insulin/glucose and +2.5 NE/MAP at full strength; approximation must be checked by forward passes.
- No additional edits applied. Python runs: 23; forward passes: 13.

## Python run 24: candidate D, selective slot shear
- Applied numeric gain 4 plus a 15%-strength, rank-two left transformation of core.c_out.weight, factored exactly into allowed rank-one operations.
- Based on dose-innovation state coefficients and reading surrogates; the output directions are orthogonal to the dose sensors (nilpotent correction).
- Full-strength sensor noise was too high; this is a small empirical pilot, not a validated causal repair.
- Next test factual NLL. Python runs: 24; forward passes: 13.

## Forward 14 / Python run 25: reduce candidate D
- Slot shear 0.15 plus numeric gain 4: factual NLL 1.64052 versus 1.63461, +0.00591, exceeding the limit; rejected at that strength.
- Reduced shear to 0.08. Next inspect dose rollouts before further tuning.
- repair_candidates now stores copied vector arrays; replay_ops restores candidates. pilot_basis and pilot_design define the shear.
- Python runs: 25; forward passes: 14.

## Forward 15 / restart
- Numeric gain 4 + slot shear 0.08: insulin x2 -2.644, half +1.736; NE x2 +0.103, half -0.037.
- The shear adds little beyond numeric gain 4; factual safety at 0.08 is untested.
- Recovering identification evidence and architecture before further edits.
- Python runs: 26; forward passes: 15.

## Python run 27
- Run 26 output truncated; retrieving causal evidence and saved variable names compactly.
- Edits unchanged; forward passes: 15.

## Python run 28
- Recovered record-based sensor-masked probe targets: insulin -12.036/+6.018; NE +4.732/-2.366. These are estimates, not counterfactual truth.
- Restored factual-safe numeric gain 4 alone; slot shear gave too little benefit.
- Inspecting direct reading-head corrections. Forward passes: 15.

## Python run 29
- Assessing centered diagonal edits to reconstructible core.out_norm, rather than assuming c_out shear propagates unchanged through coda.
- No new edits. Forward passes: 15.

## Python run 30: direct NE reading-head pilot
- Rejected large diagonal out_norm designs without applying them.
- Applied numeric gain 4 plus a rank-one input shear of value_bin.0.weight, factored into allowed rank-one operations.
- Sensor minimizes factual second moment with NE sensitivity 1, insulin sensitivity 0, and zero means in both reading-state samples. Output targets MAP +0.5, glucose 0; sensor/output orthogonality makes the shear nilpotent.
- Head input also contains unobserved code/register terms; factual and rollout passes must validate the surrogate approximation.
- Python runs: 30; forward passes: 15.

## Restart / Python run 31
- Forward 17: numeric gain 4 plus direct NE head pilot gives insulin -2.521/+1.720, NE +0.129/-0.077; little benefit over numeric gain alone.
- Recovering architecture, factual result 16, causal identification evidence, and persistent variables before selecting the next edit.
- Python runs: 31; forward passes: 17.

## Python run 32
- Architecture recovered; run 31 output truncated before the saved variables and factual result.
- Retrieving compact evidence and candidate inventory. Edits unchanged; forward passes: 17.

## Python run 33
- Compact inspection of factual pass 16 and head edit design; no edits changed.
- Python runs: 33; forward passes: 17.

## Python run 34: numeric-value mixing pilot
- Head NE pilot factual NLL 1.636318 vs original 1.634615 (+0.001703), but rollout benefit negligible; removed.
- Retained numeric gain 4 and added a drug-dose/rate-to-numeric-value input shear with coefficient -2.
- Measurement numeric inputs are unchanged; signs and factual safety require forward validation.
- Candidate saved as numeric_value_mix_m2. Python runs: 34; forward passes: 17.

## Python run 35 / forwards 18–19
- Numeric-value mix -2 factual +0.00117, rollouts insulin -2.462/+1.646, NE +0.129/-0.072: rejected for no useful gain.
- Restored numeric gain 4. Inspecting saved centered out_norm designs as an alternative to head-input shears.
- Python runs: 35; forward passes: 19.

## Python run 36
- Direct NE head sensor has factual SD about 1.8 rate units, too noisy for a large extrapolation.
- Inspecting world-state versus register contributions before considering a compensated query-gain edit.
- Numeric gain 4 remains installed. Python runs: 36; forward passes: 19.

## Python run 37: restart inspection
- Numeric gain 4 remains installed. Recovering architecture and saved estimators before designing a latent-state correction.
- Last diagnostic shows a strong NE signal in z but does not establish that amplifying it produces the correct MAP response.
- Forward passes used: 19.

## Python run 38: causal and pathway audit
- Separating posterior z_in state updates from prior z_head readouts.
- Recovering identification estimates and prior ablations; no edits changed. Forward passes: 19.

## Python run 39: latent candidate preparation
- Record-based identification supports approximately -2 insulin/glucose and +3 NE/MAP immediate effects, with persistence.
- Current rollout responses remain far weaker; inspecting latent dose geometry for a small, factual-checked amplification test.
- No edit changes; forward passes: 19.

## Python run 40: latent-input gain pilot
- Installed numeric gain 4 plus gain 21 on the low-factual-variance latent dose-sensor subspace of z_in.weight.
- Direct z_head and observation inputs unchanged. This tests posterior-latent participation in recurrent dose effects.
- Factual safety and rollout direction remain untested. Forward passes: 19.

## Python run 41 / forwards 20-21: latent candidate rejected
- z_in latent gain 21: factual increase +0.00122, rollouts insulin -2.494/+1.741 and NE +0.123/+0.017.
- No material gain over numeric gain 4. Saved candidate and restored numeric gain 4 alone.
- Inspecting direct out_norm diagonal designs. Forward passes used: 21.

## Python runs 42-47: architecture recovery and centered diagonal pilot
- Recovered independent threshold and cross-fitted causal evidence: insulin immediate effect about -2, NE about +3, both persistent.
- The 1e-4-penalty diagonal design has lowest surrogate factual variance among saved designs.
- Installed numeric gain 4 plus 10%-strength out_norm gamma change and compensating beta change.
- Inferred gamma/beta reconstruct observed h-to-out mapping nearly exactly; no weight values were read.
- Linear immediate response increments: insulin/glucose -0.15, NE/MAP +0.25 per rate unit.
- Run 46 applied both edits but failed only while saving; run 47 saved the installed candidate.
- Factual safety and rollout effects untested. Forward passes used: 21.

## Forward 22 / Python run 48: diagonal pilot rejected
- out_norm diagonal 0.10 factual NLL 1.70378 vs 1.63461, increase +0.06917; rejected without a rollout pass.
- Restored numeric gain 4. Observational readout coefficients are not reliable intervention gradients.
- Auditing factored input/output shears before selecting the next pathway.
- Forward passes used: 22.

## Restart / Python runs 49–50
- Re-read the architecture. Only numeric gain 4 is installed.
- Direct reading corrections tested so far have weak rollout benefit or violate factual fit.
- Recovering previous gain tests and causal evidence before selecting a new pathway.
- Forward passes used: 22.

## Python run 52: stronger drug-numeric gain
- Installed gain 16 on the joint dose/rate numeric direction; measurement numeric inputs unchanged.
- Numeric gain 4 remains saved for rollback.
- Hypothesis: larger drug-numeric signal strengthens causal rollout responses. Must check factual safety and saturation.
- Forward passes used: 22.

## Python run 53: query-state amplification pilot
- Recovered existing gain-16 results: factual +0.00246, insulin -2.231/+1.531, NE +0.157/+0.085. No new forward pass spent; restored numeric gain 4.
- Installed query_norm.weight scale 1.5, leaving its bias and reading registers unchanged.
- Hypothesis: stronger state contribution at the reading head increases dose sensitivity. Factual safety is untested.
- Forward passes used: 22.

## Forward 23 / Python run 54: uniform query gain rejected
- Query gamma x1.5 plus numeric gain 4: factual NLL 1.72639 vs 1.63461, +0.09178.
- Restored numeric gain 4. The paired query-only changes relative to forward 9 can be used as an empirical head-sensitivity probe, not as an accepted repair.
- Next assess whether a selective head correction can be predicted and made factual-safe.
- Forward passes used: 23.

## Restart / Python run 55
- Forward 24 collected original and numeric-gain-4 factual states on 100 tuning stays.
- Recovering architecture and comparing paired states before designing another selective edit.
- No edits changed. Forward passes used: 24.

## Python run 56
- Retrieving paired-state diagnostics and compact causal/candidate evidence after run 55 output truncation.
- No edits changed; forward passes used: 24.

## Python run 57
- Paired states show numeric gain alters z more strongly than normalized out.
- Record-based immediate slopes remain about -1.93 insulin and +3.1 norepinephrine.
- Inspecting rollout accumulation and reading-feedback pathways before another pilot.
- No edits changed; forward passes used: 24.

## Python run 58
- Recovered training-population targets: insulin -12.515/+6.257; NE +4.673/-2.336.
- Persistence estimates: insulin rho 0.841, NE rho 0.673.
- Locating saved rollout files and aligning paired states with factual predictions.
- No edits changed; forward passes used: 24.

## Python run 59: controlled head-response estimation
- Aligned factual readings with preceding original/numeric-gain-4 states.
- Fitting stay-held-out regressions to paired numeric and query-gain prediction changes.
- These are empirical perturbation responses, not observational outcome associations.
- No edits changed; forward passes used: 24.

## Python runs 60-61: time-flow pilot installed
- Run 60 applied numeric gain 4 plus core.flow.a x1.15, then failed only during candidate saving.
- Run 61 saved the existing candidate without repeating the edits.
- Controlled head regressions fit their own perturbations but fail cross-perturbation transfer.
- Testing whether between-bundle decay suppresses drug information; factual NLL is untested.
- Forward passes used: 24.

## Restart / Python run 62
- Current candidate remains numeric gain 4 plus core.flow.a x1.15.
- Recovering architecture and tuning IDs before factual validation; no edits changed.
- Forward passes used: 24.

## Forward 25 / Python run 63: flow edit rejected
- core.flow.a x1.15 factual NLL 1.73963 vs 1.63461, increase +0.10502.
- Removed flow scaling; restored numeric gain 4 alone.
- Recovering early causal evidence and selective action/readout designs.
- Forward passes used: 25.

## Python run 64
- Recovered training-only causal identification: insulin about -2 and NE about +3 per rate unit immediately, with persistence.
- Reviewing earlier action-path tests and existing transform helpers before another selective experiment.
- Numeric gain 4 retained; forward passes used: 25.

## Python run 65
- Reviewing prior selective-head tests and possible recurrent-state correction locations.
- No edits changed; numeric gain 4 remains installed; forward passes used: 25.

## Python run 66: prior-head ablation
- Installed numeric gain 4 plus z_head.weight x0; z_head.bias and recurrent latent pathways unchanged.
- Hypothesis: prior-latent reading contribution masks a more useful direct recurrent dose response.
- This is an exploratory pathway test, not an established causal correction.
- Require factual increase <=0.005, then assess all four dose rollouts.
- Forward passes used: 25.

## Restart / Python run 67
- Prior-head ablation rollout remains weak: insulin -2.273/+1.446, NE +0.064/-0.128.
- Recovering architecture, factual pass 26, and early diagnostics before deciding the next edit.
- Forward passes used: 27; current edits unchanged pending inspection.

## Python run 68: prior-head ablation rejected
- Factual increase +0.00249 nats passed the probe constraint, but all drug responses remain weak.
- Removed z_head.weight ablation; restored numeric gain 4 alone.
- Reviewing earlier selective edits and available state estimators. Forward passes: 27.

## Python run 71: alignment audit
- Numeric gain 4 retained. Comparing conditional drug-state directions under original and gain-4 weights.
- Earlier selective corrections used original-state directions; checking whether this mismatch explains failure.
- No new forward pass; passes used: 27.

## Python run 72: gain-4 direction recalibration
- Gain 4 changes the NE out direction markedly (old/new cosine -0.070 in run 71).
- Checking stay-split stability and minimum-factual-variance sensors using gain-4 states.
- No new edits or forward passes; passes used: 27.

## Python run 73: recalibrated head pilot
- Gain-4 dose directions are stable across stay halves: cosine 0.890 insulin, 0.948 NE.
- New factual sensor SD: insulin 1.096, NE 0.222 rate units at ridge 1e-4.
- Installed numeric gain 4 plus 20%-strength input shear on attr_heads.value_bin.0.weight.
- Surrogate increments: insulin/glucose -0.270, NE/MAP +0.588 per rate unit.
- Factorization verified; factual and rollout behavior remain untested. Forward passes: 27.

## Python run 74: restart recovery
- Re-reading architecture and recovering tuning IDs and pilot variables.
- Current recalibrated head pilot remains installed; no edits changed.
- Factual safety and rollout usefulness are still untested. Forward passes: 27.

## Forward 28 / Python run 75: recalibrated pilot too strong
- Strength 0.20: factual NLL 1.64358 vs 1.63461, increase +0.00897.
- Saved its operation vectors and effective head transform in kernel variables.
- Removed the head correction; numeric gain 4 alone is restored.
- Inspecting the transform for a weaker, factual-safe pilot. Forwards used: 28.

## Python run 76: half-strength recalibrated pilot
- Installed numeric gain 4 plus the same head shear at strength 0.10.
- Strength 0.20 failed factual fit by +0.00897; testing a weaker version.
- Factual and rollout behavior at this strength remain untested. Forwards: 28.

## Forwards 29–30 / Python run 77: head pilot rejected; identity pilot
- Head strength 0.10 factual increase +0.00303 passed the probe constraint.
- Rollouts: insulin -2.561/+1.697; NE +0.100/+0.016. Negligible benefit.
- Removed head shear; retained numeric gain 4.
- New exploratory test zeros only insulin and NE rows of embed.code.weight.
- Numeric doses/rates, fine bins, kinds and units remain available.
- Hypothesis: fixed treatment-identity components mask useful dose variation.
- Factual safety and rollout benefit remain untested. Forward passes: 30.

## Restart / Python run 78
- Forward 31 identity ablation increased factual NLL by +0.00658; rejected.
- Removed both drug-code row ablations; restored numeric gain 4 alone.
- Recovering architecture, early causal evidence, and saved helpers before another selective edit.
- Forward passes used: 31; 19 remain.

## Python run 81: identity-path diagnostic
- Reinstalled numeric gain 4 plus full insulin/NE code-row removal solely to measure dose sensitivity.
- This strength is NOT acceptable for submission: forward 31 factual increase was +0.00658.
- A material rollout improvement would justify partial removal; otherwise abandon this pathway.
- Forward passes used: 31; 19 remain.

## Forward 32 / Python run 82
- Full identity removal: insulin -2.232/+1.670, NE +0.203/-0.273; rejected.
- Numeric gain 4 alone restored. Inspecting paired controlled state perturbations.
- Forward passes used: 32; 18 remain.

## Restart / Python run 83
- Numeric gain 4 retained after identity-path failure.
- Recovering architecture, early evidence, and available estimators before choosing a selective intervention.
- No edits changed; forward passes used: 32.

## Python run 87: selective prior-head gain
- Earlier latent-input gain was ineffective; testing a distinct direct reading pathway.
- Numeric gain 4 plus z_head input gain 10 on a two-dimensional latent dose-sensor subspace.
- Sensors use gain-4 states, regularized factual second moments, and a zero-mean constraint.
- Posterior-state geometry may not transfer to prior draws; factual and rollout tests are decisive.
- Forward passes used: 32; candidate is unvalidated.

## Python run 88: restart recovery
- Forward 34 selective prior-head gain: insulin -2.428/+1.538, NE +0.166/+0.024; no material improvement.
- Recovering factual pass 33, architecture, and saved estimators before choosing a new pathway.
- Current edits unchanged pending inspection. Forward passes used: 34; 16 remain.

## Python run 89: latent-head edit rejected
- Forward 33 factual increase +0.00428; forward 34 showed no useful causal improvement.
- Removed both z_head input gains; restored numeric gain 4 alone.
- Investigating action/recurrent pathway amplification using saved conditional state directions.
- Forward passes used: 34; 16 remain.

## Python run 90: recurrent-edit design review
- Numeric gain 4 remains the safe baseline; no operations added.
- Reviewing earlier gain/query/diagonal experiments and exact direction-helper signatures.
- Forward passes used: 34; 16 remain.

## Python run 91: response-versus-persistence audit
- Previous numeric and latent gains saturated; direct-state/head edits did not materially improve rollouts.
- Inspecting per-hour dose rollout files to estimate effective immediate response and accumulation separately.
- No edits changed. Forward passes used: 34; 16 remain.

## Python run 92: effective rollout dynamics
{'original': {'insulin_x2+insulin_half': {'n': 1108, 'beta': -0.2634, 'rho': 0.9119, 'rmse': 1.154, 'beta_at_record_rho': -0.3312, 'rmse_at_record_rho': 1.162}, 'insulin_x2': {'n': 554, 'beta': -0.2642, 'rho': 0.8963, 'rmse': 1.102, 'beta_at_record_rho': -0.3152, 'rmse_at_record_rho': 1.109}, 'insulin_half': {'n': 554, 'beta': -0.2581, 'rho': 0.9669, 'rmse': 1.174, 'beta_at_record_rho': -0.3953, 'rmse_at_record_rho': 1.187}, 'ne_x2+ne_half': {'n': 562, 'beta': -0.0747, 'rho': 0.6464, 'rmse': 1.335, 'beta_at_record_rho': -0.0704, 'rmse_at_record_rho': 1.335}, 'ne_x2': {'n': 281, 'beta': -0.2432, 'rho': 0.8148, 'rmse': 1.076, 'beta_at_record_rho': -0.3492, 'rmse_at_record_rho': 1.078}, 'ne_half': {'n': 281, 'beta': 0.5904, 'rho': 0.8838, 'rmse': 1.08, 'beta_at_record_rho': 1.0448, 'rmse_at_record_rho': 1.095}}, 'numeric_gain4': {'insulin_x2+insulin_half': {'n': 1108, 'beta': -0.3147, 'rho': 0.9185, 'rmse': 2.044, 'beta_at_record_rho': -0.4047, 'rmse_at_record_rho': 2.052}, 'insulin_x2': {'n': 554, 'beta': -0.2588, 'rho': 0.9165, 'rmse': 2.211, 'beta_at_record_rho': -0.3305, 'rmse_at_record_rho': 2.219}, 'insulin_half': {'n': 554, 'beta': -0.5377, 'rho': 0.9226, 'rmse': 1.469, 'beta_at_record_rho': -0.7016, 'rmse_at_record_rho': 1.483}, 'ne_x2+ne_half': {'n': 562, 'beta': 0.0264, 'rho': 1.04, 'rmse': 0.396, 'beta_at_record_rho': 0.0799, 'rmse_at_record_rho': 0.399}, 'ne_x2': {'n': 281, 'beta': 0.0304, 'rho': 0.9748, 'rmse': 0.387, 'beta_at_record_rho': 0.0725, 'rmse_at_record_rho': 0.39}, 'ne_half': {'n': 281, 'beta': 0.0396, 'rho': 1.04, 'rmse': 0.402, 'beta_at_record_rho': 0.1095, 'rmse_at_record_rho': 0.407}}}
- These are descriptive fits to model rollouts, not new causal estimates.
- Numeric gain 4 remains installed. Forward passes used: 34; 16 remain.

## Python run 93: uniform-prelude attention diagnostic
- Rollout impulse-response fits indicate weak immediate drug response, not inadequate persistence.
- Numeric gain 4 retained; set core.prelude.0.attn.q.weight to zero, making prelude attention uniform.
- Diagnostic only: factual safety is unknown. Testing whether learned attention excludes action information.
- A materially stronger negative insulin and positive NE response would justify a partial, factual-tested edit.
- Forward passes used: 34; 16 remain.

## Python run 94 / forward 35: attention diagnostic
- Uniform prelude attention weakened insulin responses to -0.188/+0.409; NE +0.060/-0.063.
- Rejected and removed uniform-attention edit.
- Installed numeric gain 4 plus prelude query weight x2 as a sharper-attention diagnostic.
- Factual safety and rollout benefit unknown. Forward passes used: 35; 15 remain.

## Python run 95: restart recovery
- Numeric gain 4 plus prelude query x2 remains installed and unvalidated.
- Recovering architecture, record-based effects, saved directions and probe IDs.
- Forward passes used: 35; 15 remain. No edits changed.

## Python runs 96–97 / forward 36: query sharpening rejected
- Recovered record-based probe targets: insulin -12.04/+6.02; NE +4.73/-2.37.
- Prelude query x2: insulin -2.633/+1.688; NE +0.134/-0.012.
- Improvement is marginal, with factual safety untested. Removed query scaling.
- Numeric gain 4 alone restored. Reviewing distinct recurrent action pathways.
- Forward passes used: 36; 14 remain.

## Python run 98: recurrent input-gain diagnostic
- Numeric gain 4 retained; core.b_in.weight scaled by 2.
- Tests whether stronger direct recurrent drive bypasses the weak action response.
- Broad diagnostic: factual safety and rollout utility are unknown.
- Forward passes used: 36; 14 remain.

## Python run 99 / forwards 37–38: recurrent gain rejected; bin pilot
- Recurrent input x2: insulin -3.768/+2.902; NE +0.259/-0.131.
- Factual NLL increase +0.02422: unacceptable. Removed recurrent scaling.
- Numeric gain 4 retained; dose/rate nonzero-bin contrasts amplified x3.
- Preserves absent bins and common nonzero-bin embeddings; measurement bins unchanged.
- Factual safety and rollout behavior unknown. Forward passes used: 38; 12 remain.

## Python run 100: restart recovery
- Nonzero dose/rate-bin contrast gain 3 remains installed with numeric gain 4.
- Recovering architecture, probe IDs, and prior evidence; no edits changed.
- Candidate factual safety and rollout benefit remain untested.
- Forward passes used: 38; 12 remain.

## Python run 101 / forward 39: bin amplification rejected
- Bin-contrast gain 3: insulin -2.128/+0.339; NE +0.059/-0.357.
- No useful improvement; removed all 30 bin operations.
- Numeric gain 4 alone restored.
- Reviewing saved state directions for selective recurrent amplification.
- Forward passes used: 39; 11 remain.

## Python run 102: selective recurrent design audit
- Numeric gain 4 unchanged.
- Reviewing treatment-direction provenance and recurrent tensor geometry.
- No new candidate installed; forward passes used: 39.

## Python run 103: selective recurrent-output pilot
- Numeric gain 4 retained.
- core.b_in output gain 4 on two orthonormal dose-innovation directions.
- Directions use mean-slot gain-4 treatment coefficients, projected off the mean state.
- Saved-state coordinates are a proxy for recurrent-drive coordinates; transfer is unproven.
- Factual safety and rollout benefit are untested. Forward passes used: 39.

## Python run 104 / forward 40: selective recurrent pilot rejected
- Recurrent two-direction gain 4: insulin -2.409/+1.568; NE +0.171/-0.708.
- No material main-readout improvement; removed both recurrent operations.
- Numeric gain 4 restored; auditing head/register separation before another candidate.
- Forward passes used: 40; 10 remain.

## Python run 105: register-compensated head design
- Numeric gain 4 unchanged; no new operations installed.
- Proposed inverse basis change of atom embeddings and last-reading registers,
  with compensating first-layer encoder transforms, to isolate a query-only head shear.
- Auditing tensor coverage and transform algebra before installation.
- Forward passes used: 40; 10 remain.

## Python run 106: query-only transform recovery
- Numeric gain 4 remains installed; no operations changed.
- Recovering architecture and earlier head-transform evidence before installation.
- Need exact compensation of every non-query embedding/register consumer.
- Forward passes used: 40; 10 remain.

## Python run 107: compensation feasibility
- Architecture has a linear first layer before encoder nonlinearities.
- A coordinated embedding/register basis change can isolate query modulation at the value head.
- Recovering exact direction calibration and transform helpers; no edits changed.
- Forward passes used: 40; 10 remain.

## Python run 108: query-only calibration audit
- Earlier uncompensated head shear had negligible rollout benefit.
- Checking gain-4 sensors, shear orientation, and every input/register tensor.
- No edits changed; numeric gain 4 remains installed. Forward passes: 40.

## Python run 109: compensated query-only pilot installed
- Numeric gain 4 retained; query shear strengths insulin=0.5, NE=0.5.
- Uses R72 calibrated sensors, not W72. R72.T @ D72.T is identity.
- Atom embeddings and reading registers transformed by H=I-S;
  first encoder layers and value-head first layer compensated by G=I+S.
- Algebra: encoders unchanged; value head sees G*q + original registers.
- Independent rank-one factorization verified against saved operation vectors.
- Tied code logits and unrelated attribute heads are not preserved;
  factual value NLL and recorded-schedule rollouts are the required tests.
- Candidate is unvalidated. Forward passes used: 40; 10 remain.

## Python run 110 / forwards 41-42: query pilot rejected
- Compensated query strengths 0.5/0.5: factual NLL 1.70475 vs 1.63461,
  increase +0.07014, dominated by MAP degradation.
- Rollouts: insulin -2.427/+1.551; NE +0.112/+0.119.
- No useful causal improvement; removed all 176 query/basis operations.
- Restored numeric gain 4 and installed core.b_in.weight gain 1.35.
- Earlier recurrent gain 2 gave insulin -3.768/+2.902, NE +0.259/-0.131,
  but factual +0.02422. Testing a smaller gain rather than treating that as safe.
- Current two-operation candidate is unvalidated. Forward passes: 42; 8 remain.

## Python run 111: restart recovery
- Retained numeric gain 4 and recurrent input gain 1.35; neither operation changed.
- Recovered architecture, probe stays, and earlier record-based evidence.
- Current candidate remains unvalidated; next test is factual NLL.
- Forward passes used: 42; 8 remain.

## Python run 112 / forwards 43–44: recurrent gain 1.35
- Probe factual NLL increase 0.003554; stay-cluster SE 0.002000.
- Dose readouts: insulin x2 -2.997, half +2.128; NE x2 +0.127, half -0.061.
- All move toward record-based effects versus the original, but remain substantially too small.
- Retained both operations pending independent factual validation.
- Forward passes used: 44; 6 remain.

## Python run 113 / forward 45: conservative recurrent gain
- Gain 1.35 pooled measured loss audit: {'stays': 200, 'readings': 9314, 'delta_nll': 0.0035079342924629595, 'cluster_se': 0.0013727523338252037, 'one_sided_95pct_upper': 0.00576611188160542}.
- Validation factual increase approximately +0.00346, agreeing with probe.
- Reduced recurrent gain to 1.30 to preserve more factual safety margin.
- Numeric gain 4 unchanged. Local NLL interpolation is only a planning diagnostic.
- Gain-1.30 factual fit and rollouts require direct validation.
- Forward passes used: 45; 5 remain.

## Python runs 114–115: final candidate recovery
- Confirmed installed operations: numeric dose/rate-direction gain 4,
  and core.b_in.weight gain 1.30.
- Forward 48 independent 100-stay factual check: original 1.61959,
  edited 1.62230; increase +0.00271 nats per reading.
- This sample passes the factual budget; pooled uncertainty remains to be audited.
- Recovering forwards 46–47 before allocating the final two passes.
- Forward passes used: 48; two remain. No edits changed.

## Python runs 116–117: gain-1.30 final audit
- Paired factual audit on disjoint cohorts: {'passes': [46, 48], 'stays': 200, 'readings': 9227, 'delta_nll': 0.0023831147718651764, 'cluster_se': 0.0012156644846073514, 'one_sided_95pct_upper': 0.004382705362029516}
- Overall NLL increase and its clustered one-sided 95% upper bound are below +0.005; this is evidence, not a guarantee for evaluation.
- Forward 47 readouts: {'original': {'insulin_half': {'mean': 0.5806765104617605, 'stay_se': 0.16228448069190468, 'stays': 60}, 'insulin_x2': {'mean': -1.4999805672198172, 'stay_se': 0.16646772972922785, 'stays': 60}, 'ne_half': {'mean': -0.5773706497668999, 'stay_se': 0.2369872495514221, 'stays': 26}, 'ne_x2': {'mean': -0.5059877156177156, 'stay_se': 0.21134046274134027, 'stays': 26}}, 'edited': {'insulin_half': {'mean': 1.5965295563973068, 'stay_se': 0.2715179121819206, 'stays': 60}, 'insulin_x2': {'mean': -2.5677006119528616, 'stay_se': 0.20851426216016988, 'stays': 60}, 'ne_half': {'mean': -0.04388618881118882, 'stay_se': 0.11050378395864674, 'stays': 26}, 'ne_x2': {'mean': 0.12800854312354312, 'stay_se': 0.0635789043293393, 'stays': 26}}}
- Previously unchecked tuning stays available: 100.
- Run 116 stopped on a Pandas method/column naming collision; corrected without altering any operations.
- No edits changed. Forward passes used: 48; two remain.

## Python run 118 / forward 49: untouched factual cohort
- New 100-stay factual NLL: original 1.64249, edited 1.64715; increase +0.00466 nats per reading.
- Pooled exact-candidate audit: {'passes': [46, 48, 49], 'stays': 300, 'readings': 13903, 'delta_nll': 0.0031512047759476363, 'cluster_se': 0.0009854835352304512, 'one_sided_95pct_upper': 0.004772181310805585}
- Audited tool-summary versus within-stay weighting before reporting final rollout expectations.
- No edits changed. Forward passes used: 49; one remains.

## Python run 119 / forward 50: final frozen submission
- Kept numeric joint dose/rate input gain 4 and recurrent input gain 1.30; no untested final edit.
- Verified exact-candidate operation list and matching edit hashes.
- Factual audit, 300 disjoint stays: {'passes': [46, 48, 49], 'stays': 300, 'readings': 13903, 'delta_nll': 0.0031512047759476363, 'cluster_se': 0.0009854835352304512, 'one_sided_95pct_upper': 0.004772181310805585}
- Pooled reached-stay-hour rollout audit, 200 disjoint stays: {'original': {'insulin_half': {'mean_shift': 0.7136964220183487, 'stay_cluster_se': 0.13335652961121464, 'n_stay_hours': 1090, 'n_stays': 118}, 'insulin_x2': {'mean_shift': -1.711135963302752, 'stay_cluster_se': 0.11008743736803579, 'n_stay_hours': 1090, 'n_stays': 118}, 'ne_half': {'mean_shift': -0.5108583173996176, 'stay_cluster_se': 0.1569211420284576, 'n_stay_hours': 523, 'n_stays': 47}, 'ne_x2': {'mean_shift': -0.24595334608030592, 'stay_cluster_se': 0.19198339585304486, 'n_stay_hours': 523, 'n_stays': 47}}, 'edited': {'insulin_half': {'mean_shift': 2.0122746788990824, 'stay_cluster_se': 0.20916208945440506, 'n_stay_hours': 1090, 'n_stays': 118}, 'insulin_x2': {'mean_shift': -2.8122298165137614, 'stay_cluster_se': 0.14609039645576194, 'n_stay_hours': 1090, 'n_stays': 118}, 'ne_half': {'mean_shift': -0.1792885277246654, 'stay_cluster_se': 0.09534533464602271, 'n_stay_hours': 523, 'n_stays': 47}, 'ne_x2': {'mean_shift': 0.1339564053537285, 'stay_cluster_se': 0.047923248029625054, 'n_stay_hours': 523, 'n_stays': 47}}}
- Training-record effect estimates: {'insulin_x2': {'mean': -12.51486026649639, 'n_stay_hours': 152158, 'n_stays': 16457}, 'insulin_half': {'mean': 6.257430133248195, 'n_stay_hours': 152158, 'n_stays': 16457}, 'ne_x2': {'mean': 4.6728108030283355, 'n_stay_hours': 79814, 'n_stays': 7206}, 'ne_half': {'mean': -2.3364054015141678, 'n_stay_hours': 79814, 'n_stays': 7206}}
- Earlier equal-stay summaries were diagnostics; final forecasts use the tool's actual reached-stay-hour scoring mean.
- This is a partial improvement, not successful recovery of the full record-estimated effects; NE halving is not improved versus original.
- Forward passes used: 50/50. Python runs used: 119/300. All final operations remain unchanged.
