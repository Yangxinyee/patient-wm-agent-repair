# Weight-repair notebook

## Run 1: initial inspection
- Read architecture, data schema, tensor shapes, vocabulary, and a record sample.
- No drug-effect conclusion or weight edit yet.
- All causal conclusions will come from the supplied records.
- Python runs: 1; forward passes: 0.

## Run 2: records and vocabulary
- Architecture has a dedicated action encoder; this is a candidate edit boundary, not evidence of drug effects.
- Loaded all training and tuning records and inspected event timing and dose distributions.
- Python runs: 2; forward passes: 0.

## Run 3: hourly panel and preliminary associations
- Doses occur at hour + 0.0833, after same-hour measurements; next-hour outcomes are post-treatment.
- Constructed complete hourly histories, with zero rates for absent infusion records.
- Computed raw and flexible history-adjusted one-hour slopes; these remain provisional because treatment and observation selection may confound them.
- Preliminary estimates: {'glucose_finger': {'n': 298450, 'beta': array([-1.9552787 , -0.00447367]), 'se': array([0.04175153, 0.27588974]), 'residual_dose_sd': array([0.62089111, 0.08481478])}, 'glucose_lab': {'n': 28483, 'beta': array([-1.61143064,  0.23353193]), 'se': array([0.10999966, 0.47828202]), 'residual_dose_sd': array([0.44313316, 0.08471139])}, 'map_arterial': {'n': 312563, 'beta': array([-0.01420961,  2.50709245]), 'se': array([0.01662592, 0.08518661]), 'residual_dose_sd': array([0.46802037, 0.10263334])}, 'map_cuff': {'n': 333382, 'beta': array([0.00668627, 0.46524658]), 'se': array([0.028022  , 0.21911062]), 'residual_dose_sd': array([0.46792872, 0.06450492])}}
- Python runs: 3; forward passes: 0.

## Run 4 / forwards 1–2: baseline and dose-policy models
- Original tuning-100 rollout: insulin x2 -2.016; insulin half +1.051; NE x2 +0.309; NE half -0.070.
- Original factual value NLL: 1.63155 over 4,751 readings.
- Fit two-fold, stay-held-out flexible dose-policy models using only pre-dose histories.
- These models are for identification diagnostics, not weight training.
- Python runs: 4; forward passes: 2.

## Run 5: cross-fitted next-hour effects
- Dose-policy out-of-stay R2: insulin 0.8660, NE 0.9351; residual SD 0.3678 and 0.07559 respectively.
- Used two-fold, stay-held-out nonlinear outcome models and clustered residual-on-residual regressions.
- Checked sensor, positive-dose, and early/late subgroups; estimates still require confounding and dynamics checks.
- DML results: {'glucose_finger': {'beta': array([-1.93474217,  0.33946561]), 'se': array([0.04454258, 0.26229829]), 'n': 400154}, 'glucose_lab': {'beta': array([-2.01483673, -0.33459203]), 'se': array([0.0747496 , 0.35865811]), 'n': 143791}, 'map_arterial': {'beta': array([0.01574776, 3.05259229]), 'se': array([0.02054298, 0.0887042 ]), 'n': 328940}, 'map_cuff': {'beta': array([-0.03133786,  2.52816442]), 'se': array([0.02915835, 0.17344546]), 'n': 439865}}
- Python runs: 5; forward passes: 2.

## Run 6: dynamic identification
- Cross-fitted immediate effects: glucose finger -1.9347 (SE .0445), lab -2.0148 (.0747); arterial MAP +3.0526 (.0887), cuff +2.5282 (.1734).
- Estimated treatment impulse responses using dose innovations and removed their induced future-dose changes by moment deconvolution.
- Used independent/randomly sampled glucose_lab for glucose dynamics and arterial MAP for pressure dynamics, with stay-cluster bootstrap.
- Impulse estimates: {'glucose_lab': array([-2.07283356, -2.23466817, -2.14515517, -1.43875875, -1.28052947,
       -0.90473374]), 'map_arterial': array([2.95159917, 2.49611187, 1.62799136, 1.02869814, 0.34279884,
       0.21392442])}
- Interpretation depends on residual treatment variation being conditionally exogenous; dose-regime checks assess linearity within observed support.
- Python runs: 6; forward passes: 2.

## Run 7: restart inspection
- Re-read architecture and schema and inventoried retained analysis objects before choosing an edit.
- No new causal conclusion or weight edit in this run.
- Python runs: 7; forward passes: 2.

## Run 8: edit-boundary inspection
- Dedicated action encoding changes a shared state, so broad edits may damage non-value outputs.
- Inspecting value-head and state layouts for a conditionally centered, targeted correction.
- Python runs: 8; forward passes: 2.

## Run 9 / forward 3: original training states
- Obtained 4,035 post-bundle states from 100 training stays.
- Checked whether linear state readouts retain conditionally centered dose innovations.
- Recovery diagnostics: {('out', 'actual rate'): [0.7464798056041226, 0.5793095632860579], ('out', 'dose innovation'): [0.014224321981154309, -0.010256190838008106], ('out+h+z', 'actual rate'): [0.8209381610756414, 0.5761447082503632], ('out+h+z', 'dose innovation'): [0.14643522829061584, -0.10865335500972084]}
- No weights edited.
- Python runs: 9; forward passes: 3.

## Run 10 / forward 4: factual residual and action-contrast diagnostics
- Original train-100 factual value NLL: 1.63805 over 4,510 readings.
- Factual-error innovation slopes: {'glucose_finger': {'beta': [-2.0213525150007543, -0.8642344520105519], 'se': [1.0749416026751106, 6.437951025950719], 'n': 1207}, 'glucose_lab': {'beta': [-2.1402151269169334, 5.8731650809484615], 'se': [2.8894259636083786, 7.18666777339531], 'n': 489}, 'map_arterial': {'beta': [-0.4176368016509025, 4.049828159080294], 'se': [0.33286110151897685, 1.9466967832560893], 'n': 914}, 'map_cuff': {'beta': [-1.3798312320807482, 3.841096696440792], 'se': [0.4707643862776628, 2.4905594022701707], 'n': 1567}}
- Pre/post-action innovation recovery: {('pre+post out', 10.0): [0.004214188536758923, -0.019954426503093314], ('pre+post out', 100.0): [0.012494810294534697, -0.002386756631993059], ('pre+post out', 1000.0): [0.006069711150727075, 0.0011565628426700725], ('action contrasts', 10.0): [0.03918410241440473, -0.13992020687565376], ('action contrasts', 100.0): [0.07106420927132506, -0.07724699273376356], ('action contrasts', 1000.0): [0.034740616853687345, -0.03114449562207211]}
- No weights edited. These small-sample diagnostics do not replace full-record causal estimates.
- Python runs: 10; forward passes: 4.

## Run 11: action/posterior reconstruction
- Small train-100 factual residuals suggest missing responses, but are not precise effect estimates.
- Reconstructed stateless action and posterior primitives; compared directly to saved z means.
- Best reconstruction configurations: {'insulin': (0.09675629436969757, 'relu', 'log1p', False, 3, 2, 'chunk'), 'norepinephrine': (0.12859904766082764, 'gelu', 'log1p', False, 2, 2, 'chunk')}
- No weights edited.
- Python runs: 11; forward passes: 4.

## Run 12: reconstruction audit
- The posterior reconstruction remains inaccurate; no edit is justified by it yet.
- Re-reading the exact architecture and inventorying retained helpers before further reconstruction.
- Python runs: 12; forward passes: 4; no weights edited.

## Runs 13–14: vocabulary and retained-state audit
- Run 13 printed vocabulary and remaining tensors but its global inventory failed on an imported callable.
- Fine-bin edges are strict (edges below x), with repeated insulin edges; reconstruction must honor this.
- Run 14 fixes the inventory and inspects saved-state metadata.
- Python runs: 14; forward passes: 4; no weights edited.

## Run 15: primitive and record-field inspection
- Saved states provide direct posterior reconstruction targets and previous-state inputs.
- Considering a conditionally centered posterior-mean edit, but only after validating reconstruction and signal isolation.
- Python runs: 15; forward passes: 4; no weights edited.

## Run 16: posterior convention search
- Tested strict/non-strict bin indexing, kq and categorical embeddings, empty-world encodings, and independent MLP activations.
- Best reconstruction candidates: {'insulin': [(2.778830094030127e-06, 'gelu', 'gelu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 3, 2, 'none')), (1.5715717381681316e-05, 'gelu', 'gelu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 3, 2, 'none')), (3.302994082332589e-05, 'gelu', 'gelu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 3, 2, 'action')), (4.291285222279839e-05, 'gelu', 'gelu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 3, 2, 'action')), (0.0023337218444794416, 'gelu', 'silu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', -1, 3, 2, 'status_action')), (0.002334222663193941, 'gelu', 'silu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', 3, 3, 2, 'status_action')), (0.0023423698730766773, 'gelu', 'silu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', -1, 3, 2, 'status_action')), (0.0023432474117726088, 'gelu', 'silu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', 3, 3, 2, 'status_action')), (0.0027627081144601107, 'silu', 'gelu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', 2, 2, 2, 'status_action')), (0.0027706173714250326, 'silu', 'gelu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', 2, 2, 2, 'status_action')), (0.002889099530875683, 'silu', 'gelu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', 2, 2, 2, 'all')), (0.0028970541898161173, 'silu', 'gelu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', 2, 2, 2, 'all'))], 'norepinephrine': [(5.276543106447207e-06, 'gelu', 'gelu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 4, 3, 'none')), (1.7188651327160187e-05, 'gelu', 'gelu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 4, 3, 'none')), (3.898631257470697e-05, 'gelu', 'gelu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 4, 3, 'action')), (4.201546107651666e-05, 'gelu', 'gelu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 4, 3, 'action')), (0.0032730870880186558, 'gelu', 'silu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 4, 3, 'all')), (0.003275417722761631, 'gelu', 'silu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 4, 3, 'none')), (0.003281942568719387, 'gelu', 'silu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 4, 3, 'all')), (0.0032842038199305534, 'gelu', 'silu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('left', 0, 4, 3, 'none')), (0.004278383683413267, 'silu', 'gelu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('right', 0, 4, 3, 'all')), (0.0042791725136339664, 'silu', 'gelu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('right', 0, 4, 3, 'all')), (0.004419019911438227, 'silu', 'gelu', 'log1p/10', 'empty_gelu', 'gelu', 'chunk', ('right', 0, 4, 3, 'none')), (0.004419801756739616, 'silu', 'gelu', 'log/10', 'empty_gelu', 'gelu', 'chunk', ('right', 0, 4, 3, 'none'))]}
- Python runs: 16; forward passes: 4; no weights edited.

## Run 17: complete posterior and centered-signal audit
- Single-start drug posterior reconstruction validated at RMS below 6e-6.
- END convention candidates: [(0.08152579516172409, 'absent', 'absent'), (0.08275412023067474, 'absent', 'unit_only'), (0.08279018849134445, 'absent', 'rate_only'), (0.09130393713712692, 'absent', 'known'), (0.09511437267065048, 'rate_old', 'absent')]
- Posterior innovation recovery: {0.1: {'all': [0.048026699362843095, -0.31104207741006396], 'insulin_active': 0.22091068641940004, 'norepinephrine_active': -0.05049994634370658}, 1: {'all': [0.08198373898787636, -0.21176427299308265], 'insulin_active': 0.2277423829284697, 'norepinephrine_active': -0.04770163732936239}, 10: {'all': [0.1571004658720564, -0.08934275919631762], 'insulin_active': 0.23413193892893558, 'norepinephrine_active': -0.03893552456782223}, 100: {'all': [0.16215398503672074, -0.034179657171851785], 'insulin_active': 0.19247210437507634, 'norepinephrine_active': -0.02523959260530262}}
- Python runs: 17; forward passes: 4; no weights edited.

## Run 18: observation and END routing audit
- Run 17's full-bundle reconstruction failed, so its innovation recovery is not used for an edit.
- Best observation conventions: [(0.0275043323636055, 0, 2, 0, 'left', 'encoded'), (0.02947436086833477, 1, 2, 0, 'left', 'encoded'), (0.03308076411485672, 3, 2, 0, 'left', 'encoded')]
- Best END conventions: [(2.119088458130136e-05, 'world', 'absent', 'absent', 0), (0.0018554319394752383, 'world', 'absent', 'absent', 1), (0.0021261335350573063, 'world', 'absent', 'absent', 2)]
- Full noninitial posterior RMS: 0.01908847875893116
- Python runs: 18; forward passes: 4; no weights edited.

## Run 19: sensor-specific reconstruction audit
- END atoms enter the world encoder with absent attributes; empty action sets are encoded rather than zero.
- Action-bundle posterior RMS now below 2e-5, but ordinary-reading inputs still need resolution.
- Single-sensor convention candidates: {'glucose_finger': [(1.0317844498786144e-06, 0, 0, 'left', 'normal'), (0.006117966026067734, 1, 0, 'left', 'normal'), (0.007006202824413776, 0, 0, 'right', 'normal'), (0.009743315167725086, 1, 0, 'right', 'normal'), (0.016004769131541252, 1, 1, 'left', 'normal')], 'glucose_lab': [(6.976764552746317e-07, 5, 0, 'left', 'normal'), (0.004495772998780012, 5, 0, 'right', 'normal'), (0.01695399358868599, 5, 1, 'left', 'normal'), (0.01726042479276657, 5, 1, 'right', 'normal'), (0.019207660108804703, 5, -1, 'left', 'normal')], 'map_arterial': [(7.991941970431071e-07, 2, 0, 'left', 'normal'), (0.01745777577161789, 2, 1, 'left', 'normal'), (0.01775841973721981, 2, -1, 'left', 'normal'), (0.020669657737016678, 2, 3, 'left', 'normal'), (0.021116411313414574, 2, 2, 'left', 'normal')], 'map_cuff': [(4.3851633790836786e-07, 2, 0, 'left', 'normal'), (0.01400642842054367, 2, -1, 'left', 'normal'), (0.014127084985375404, 2, 1, 'left', 'normal'), (0.016352832317352295, 2, 2, 'left', 'normal'), (0.016451572999358177, 2, 3, 'left', 'normal')]}
- Python runs: 19; forward passes: 4; no weights edited.

## Runs 21–22: inventory recovery
- Run 21 stopped while inspecting an imported NumPy callable; no reconstruction test or weight edit occurred.
- Run 22 restricts inspection to concrete data types and audits the observation encoding conventions.
- Python runs: 22; forward passes: 4; no weights edited.

## Runs 23–26: posterior validation and edit preparation
- Lab glucose requires unit code 5, not kq 5. The attempted kq correction failed without changing model weights.
- Posterior layers are z_post.0 and z_post.2, not core.z_post.*.
- Full noninitial posterior RMS after correcting the unit is 6.052e-5 (median 9.70e-7; p95 2.225e-5; max 7.016e-4).
- Run 25's composition-print failure occurred after successful reconstruction. Inspecting core layout and dynamic causal objects next.
- Python runs: 26; forward passes: 4; no weights edited.

## Run 27: local core reconstruction
- Full posterior reconstruction passed in run 25; its later grouping print failed because bundle entries are (record, kind) pairs.
- Core convention results (out/h RMS): {(1.0, False): (0.2553735375404358, 11.284152030944824), (1.0, True): (0.09784771502017975, 5.188462734222412), (0.1, False): (0.08613714575767517, 4.496523857116699), (0.1, True): (0.09909460693597794, 5.37593936920166)}.
- Testing a conditionally centered dose-residual correction in posterior means; no edit until the transfer path is validated.
- Python runs: 27; forward passes: 4; no weights edited.

## Run 28: restart recovery
- Core reconstruction in run 27 remains unvalidated; no edit is justified from it yet.
- Recovering architecture, prior causal evidence, and persistent kernel objects.
- Python runs: 28; forward passes: 4; no weights edited.

## Run 29: targeted restart recovery
- Architecture recovered; separate action encoder and posterior latent pathway are candidate edit boundaries.
- Recovering dynamic identification results and reconstruction objects before selecting an edit.
- Python runs: 29; forward passes: 4; no weights edited.

## Run 30: recurrent-path audit
- Record-derived impulses suggest persistent insulin effects and faster MAP-effect decay.
- Recovering exact recurrent tensor layout and existing helpers to resolve state mismatch.
- Python runs: 30; forward passes: 4; no weights edited.

## Run 31: attention and memory-age convention search
- Tested attention head counts and age scaling against saved states; no model weights changed.
- Best configurations: [(0.07499662786722183, 3.5020477771759033, 16, 0.0, False), (0.0754963830113411, 3.4932687282562256, 4, 0.0, False), (0.07576201111078262, 3.512948989868164, 8, 0.0, False), (0.07583940774202347, 3.5057852268218994, 12, 0.0, False), (0.07744330167770386, 3.554762125015259, 6, 0.0, False)]
- Python runs: 31; forward passes: 4; no weights edited.

## Run 32: exact update audit
- Head-count/age search failed to validate reconstruction; documented head count is four.
- Inspecting helper operations and input scales rather than selecting an inaccurate best fit.
- Python runs: 32; forward passes: 4; no weights edited.

## Run 33: memory convention audit
- Tested prior/current memory inclusion, memory sources and normalization, age units, and FFN activations.
- Best configurations: [(1.760388840921223e-05, 0.001290809945203364, 'world', True, 'none', 'seconds', False, 'gelu'), (0.011621134355664253, 0.49570149183273315, 'world', True, 'plain', 'seconds', False, 'gelu'), (0.015962664037942886, 0.6960696578025818, 'world', True, 'block', 'seconds', False, 'gelu'), (0.022953249514102936, 1.1055859327316284, 'world', True, 'none', 'seconds', False, 'silu'), (0.022979507222771645, 1.1062058210372925, 'world', True, 'plain', 'seconds', False, 'silu')]
- No weights edited; approximate state agreement is not treated as validation.
- Python runs: 33; forward passes: 4.

## Run 34: recurrent reconstruction resolved
- The current WORLD bundle is appended to memory BEFORE cross-attention.
- Age uses log1p(seconds), GELU; age embedding is added to both key and value inputs.
- General validated helper: repaired_core_local(h,e,u,z,prior_memory,mask,ages).
- Full 64-probe RMS: out=1.5349868e-05, h=0.0012857548.
- Candidate posterior dose features are dose_feature_coef (192x2) and dose_feature_bias (2), from original post_hidden_audit.
- Cross-validation results: [(0.07894215323415066, 0.1, [np.float64(0.16810365627480106), np.float64(-0.010219349806499745)], [0.04194379194232444, 0.005418331957492828]), (0.06945217493440081, 0.01, [np.float64(0.165801251589079), np.float64(-0.026896901720277366)], [0.05694890089996388, 0.008196133134592972]), (0.06497856057120077, 1.0, [np.float64(0.13067348422924696), np.float64(-0.0007163630868454174)], [0.02645280981870855, 0.003836276132245168]), (0.06386291275502637, 0.001, [np.float64(0.1639971682870226), np.float64(-0.03627134277696986)], [0.0647197671207628, 0.009968171918089554]), (0.04419821465809565, 10.0, [np.float64(0.0869670342183152), np.float64(0.001429395097876096)], [0.016603358854153788, 0.0030243071437987337]), (0.0364293107035788, 100.0, [np.float64(0.07227933743306214), np.float64(0.000579283974095457)], [0.013722652597279974, 0.0021343280758726376])]
- These are diagnostic statistical fits, not edits; next test off-policy feature responses and choose physiological transfer directions.
- Python runs: 34; forward passes: 4; no weights edited.

## Run 35: off-policy dose-feature assay
- Existing posterior features had active-dose CV R2 .168 (insulin) and -.010 (NE); no edit yet.
- Scaled-action helper rr35_scaled_action_encoding reconstructs factor-one actions before multiplier tests.
- Counterfactual scalar responses: {'insulin': {0.0: {'mean_feature_change': [-0.21067474782466888, 0.003608871717005968], 'mean_rate_change': -1.5648148148148149, 'own_feature_slope': 0.16174676016095055, 'own_delta_quantiles': [-0.6670417726039887, -0.024058356881141663, -0.0003611899912357361]}, 0.5: {'mean_feature_change': [-0.16906307637691498, 0.002442043973132968], 'mean_rate_change': -0.7824074074074074, 'own_feature_slope': 0.2425542240259639, 'own_delta_quantiles': [-0.6484986573457718, -0.0014237985014915466, 0.009099964052438726]}, 1.5: {'mean_feature_change': [0.1165228933095932, 0.00027755642076954246], 'mean_rate_change': 0.7824074074074074, 'own_feature_slope': 0.10830405529257241, 'own_delta_quantiles': [-0.02571732550859451, 0.0007392913103103638, 0.6735776156187057]}, 2.0: {'mean_feature_change': [0.3485238254070282, 0.001979737775400281], 'mean_rate_change': 1.5648148148148149, 'own_feature_slope': 0.125960810505725, 'own_delta_quantiles': [-0.046000044047832485, 0.5259648859500885, 0.7823596537113184]}, 3.0: {'mean_feature_change': [0.336784303188324, 0.0022918665781617165], 'mean_rate_change': 3.1296296296296298, 'own_feature_slope': 0.06075495195463642, 'own_delta_quantiles': [-0.05757236182689666, 0.5198003053665161, 0.7388702988624567]}}, 'norepinephrine': {0.0: {'mean_feature_change': [0.002092755166813731, 0.0002958528930321336], 'mean_rate_change': -0.43416434540389975, 'own_feature_slope': -0.0004505283900178966, 'own_delta_quantiles': [-0.0004786699078977108, 0.00025813281536102295, 0.0011574702803045511]}, 0.5: {'mean_feature_change': [0.0002709237451199442, 2.7596717700362206e-05], 'mean_rate_change': -0.21708217270194988, 'own_feature_slope': 0.00022719093186169593, 'own_delta_quantiles': [-0.0006400971207767725, -5.83035871386528e-05, 0.0008708017412573098]}, 1.5: {'mean_feature_change': [0.0005914452485740185, 0.00012170652917120606], 'mean_rate_change': 0.21708217270194988, 'own_feature_slope': 0.0009654526247335035, 'own_delta_quantiles': [-0.0005291061010211706, 5.771522410213947e-05, 0.001099053071811795]}, 2.0: {'mean_feature_change': [0.0044122664257884026, 0.0007710559875704348], 'mean_rate_change': 0.43416434540389975, 'own_feature_slope': 0.0025260932755774034, 'own_delta_quantiles': [-0.0005615878850221634, 0.00012414529919624329, 0.005414054635912179]}, 3.0: {'mean_feature_change': [0.023111050948500633, 0.002157476730644703], 'mean_rate_change': 0.8683286908077995, 'own_feature_slope': 0.0031339260917329103, 'own_delta_quantiles': [-0.0007757513783872127, 0.0002228235825896263, 0.010614319751039146]}}}
- Python runs: 35; forward passes: 4; no weights edited.

## Run 36: recovery after conversation restart
- Factor-one action reconstruction is exact, but posterior dose features saturate for insulin and are nearly dose-insensitive for norepinephrine.
- Recovering architecture and record-derived identification before selecting an edit pathway.
- Python runs: 36; forward passes: 4; no weights edited.

## Run 37: recover dynamic identification
- Earlier cross-fitted one-hour estimates are about -2 for insulin/glucose and +3 for norepinephrine/MAP, with sensor agreement.
- Recovering impulse-response estimates and existing helpers; no edit yet.
- Python runs: 37; forward passes: 4.

## Run 38: prepare a dose-centered posterior correction
- Record-derived insulin impulses persist for several hours; MAP impulses decay faster.
- Inspecting validated local update helpers and saved posterior inputs before implementing a correction.
- Python runs: 38; forward passes: 4; no weights edited.

## Runs 39–40: direct action-dose decoding and policy centering
- Run 39 stopped on a named-tuple diagnostic-print error before fitting; fixed in run 40.
- Multiplier action-readout held-out RMSE: [(1e-06, [0.11796039661149307, 0.028787601049422212, 0.05181061032989907, 0.04790709477529297]), (0.0001, [0.12256598536032551, 0.03786559299994172, 0.05190349729761165, 0.04801767963061905]), (0.01, [0.24398175298729258, 0.07679840512199322, 0.052575374169505536, 0.049377883287424505]), (1.0, [0.6085731199577677, 0.21816193296959493, 0.06170188105652034, 0.06500822495085234])]
- Active-dose context prediction RMSE: {'insulin': [(0.1, 0.5782961461526772), (1.0, 0.5547947512352039), (10.0, 0.5362032981303506), (100.0, 0.530997116998796), (1000.0, 0.603127566315318)], 'norepinephrine': [(0.1, 0.288115039159916), (1.0, 0.2806670976643769), (10.0, 0.2692955370780582), (100.0, 0.26784869582792753), (1000.0, 0.27970046471935706)]}
- No model tensors edited.
- Python runs: 40; forward passes: 4.

## Run 41: nonlinear, dose-invariant policy centering
- Linear context centering is inadequate, especially for norepinephrine; testing existing nonlinear posterior features with counterfactual invariance constraints.
- Diagnostics: [{'invariance_penalty': 0.0, 'active_innovation_RMSE': [0.3778736561416944, 0.24010843202892565], 'absent_drug_RMS': [0.08725812270437652, 0.040178090648147265], 'dose_response_slopes': [0.8816287384288601, 1.0216930443729768]}, {'invariance_penalty': 0.01, 'active_innovation_RMSE': [0.3732257249913067, 0.2400360115967246], 'absent_drug_RMS': [0.08778011265441323, 0.040214673639178355], 'dose_response_slopes': [0.8963582577044326, 1.0214905880392395]}, {'invariance_penalty': 0.1, 'active_innovation_RMSE': [0.3802566197991127, 0.2397184636192632], 'absent_drug_RMS': [0.09343580761226314, 0.04047269946316477], 'dose_response_slopes': [0.9554710857409977, 1.019711098502509]}, {'invariance_penalty': 1.0, 'active_innovation_RMSE': [0.44432894443626, 0.24052347671780142], 'absent_drug_RMS': [0.11182384014263769, 0.04148954711891165], 'dose_response_slopes': [1.0069218395049708, 1.0107793697628735]}, {'invariance_penalty': 10.0, 'active_innovation_RMSE': [0.5278516617037933, 0.24659584392926945], 'absent_drug_RMS': [0.13925258569282126, 0.046415878578412555], 'dose_response_slopes': [1.0138826837763941, 1.0033943348260026]}]
- No tensors edited. Python runs: 41; forward passes: 4.

## Run 42: target-alignment and prediction-readout audit
- Dose-invariant centering retains intervention slopes, but norepinephrine centering error remains large.
- Checking innovation alignment and head/register layout before committing an edit.
- Python runs: 42; forward passes: 4; no tensors edited.

## Run 43: value-head reconstruction assay
- Innovation alignment exactly matches the retained targets.
- Best value-head conventions (prior mean approximation): [(0.1450220802807591, 1, 3, False, 6.047452705431512), (0.1450220802807591, 1, 3, True, 6.047452705431512), (0.24712339659102198, 1, 7, False, 10.484804885742795), (0.24712339659102198, 1, 7, True, 10.484804885742795)]
- No tensors edited. Python runs: 43; forward passes: 4.

## Run 44: recover edit implementation context
- Prior-mean head reconstruction is approximate, not an exact forward surrogate.
- Recovering architecture, identification evidence, and retained helpers before editing.
- Python runs: 44; forward passes: 4; no tensors edited.

## Run 45: focused recovery
- Shared recurrent action pathway and last-reading feedback require dynamic calibration.
- Retrieving truncated identification notes and helper names.
- Python runs: 45; forward passes: 4; no tensors edited.

## Run 46: compact helper inventory
- Exact posterior reconstruction was previously validated; avoiding repeated broad searches.
- Inspecting only validated primitives and recent correction objects.
- Python runs: 46; forward passes: 4; no tensors edited.

## Run 47: select dose-channel surgery
- Existing posterior dose features saturate; direct action decoder covers multiple multipliers.
- Checking validated core error and record-derived aggregate targets before reserving linear dose channels.
- Python runs: 47; forward passes: 4; no tensors edited.

## Run 54: bin-table correction
- Added the absent-value centre to the 16 fine-bin centres; testing local readouts before weight surgery. No weights edited.
- Python runs: 54; forward passes: 4.

## Run 56: posterior response assay
- Corrected readout bin dimensions and aligned probe registers.
- Measured finite-difference responses of all 32 posterior coordinates and fitted two small, regularized drug-response directions.
- No weights edited. Python runs: 56; forward passes: 4.

## Runs 55–58: causal pathway audit
- Training-record one-hour effects: insulin/glucose about -2 per rate unit; NE/arterial MAP about +3.05. Glucose impulses persist longer than MAP.
- Existing posterior-coordinate directions are too weak/heterogeneous for a safe direct edit.
- Exact reconstructed value-head hidden features. Complete-path immediate slopes on 64 probes: insulin x2 -0.342 (finger), x3 -0.178; NE x2 +0.249 (arterial).
- No weights edited. Expanding state coverage before a preservation-constrained head correction.
- Python runs: 58; forward passes: 4.

## Run 59: preservation-constrained head dataset
- Reconstructed all 100 training stays; checked core against validated probes.
- Built exact value-head features and local dose counterfactuals at multipliers 0, 0.5, 1.5, 2, and 3, without modifying weights.
- Python runs: 59; forward passes: 4.

## Run 60: recover preservation-fit interfaces
- Expanded core matches validated probes; 4,035 factual feature rows and 4,000 counterfactual rows retained.
- Inspecting exact head layout and dataset interfaces before a preservation-constrained, non-gradient fit.
- Python runs: 60; forward passes: 4; no tensors edited.

## Run 61: focused dataset inspection
- Architecture confirms value-head-only edits leave other factual heads directly unchanged.
- Previous broad inventory was truncated; inspecting newest tables and reconstruction helpers only.
- Python runs: 61; forward passes: 4; no tensors edited.

## Run 63: preservation-constrained head frontier
- Run 62 inspection stopped on a formatting-only ndarray/tensor error; no edits occurred.
- Fit analytical rank-one logit-tilt corrections, holding out stay IDs divisible by five.
- Evaluated exact reweighted categorical probabilities, factual NLL, and local effects at five multipliers.
- Diagnostics retained in diagnostics63; coefficients in b63_candidates. No weights edited.
- Python runs: 63; forward passes: 4.

## Run 64: reject head-only correction
- Rank-one head frontier cannot approach identified drug slopes within factual-fit budget.
- Penalty 1 held-out dNLL -0.000555 but insulin x2 slope only -0.371 and NE x2 +0.012.
- Stronger corrections damage factual NLL and still miss NE; no candidate submitted.
- Investigating dose-centered posterior/state correction using record-based innovations.
- Python runs: 64; forward passes: 4; no tensors edited.

## Run 65: prior-head response assay
- Posterior response directions are weak and require large changes; testing prior-to-head route instead.
- Measured numerical prior-mean sensitivities over mixed insulin/NE probes and fitted analytical response directions.
- J65 and D65_candidates retained; all scratch weights restored. No submitted edits.
- Python runs: 65; forward passes: 4.

## Run 66: explicit dose-linear posterior features
- Prior-head sensitivities are substantially stronger than posterior-update sensitivities.
- Existing state probes retain little dose innovation, motivating an explicit carrier.
- Reserved four low-importance posterior hidden rows for GELU +/- pairs decoding the two rates.
- Refit original posterior outputs and dose-centered readouts analytically; candidates retained, not submitted.
- Python runs: 66; forward passes: 4; no tensors edited.

## Run 67: restart recovery
- Carrier features preserve posterior outputs locally, but factual safety and dynamic effects remain untested.
- Recovering architecture, original identification evidence, and candidate interfaces before an actual edit.
- Python runs: 67; forward passes: 4; no submitted edits.

## Run 68: focused recovery
- Architecture and original baseline recovered; insulin and NE rollouts underrespond to record-derived effects.
- Retrieving dynamic targets and explicit carrier/prior correction objects after output truncation.
- Python runs: 68; forward passes: 4; no submitted edits.

## Run 69: response-route recovery
- Explicit posterior carrier is available; inspecting validated full-path and prior-head sensitivities.
- Next step is a small centered candidate and an external factual/rollout test, not further unconstrained fitting.
- Python runs: 69; forward passes: 4; no submitted edits.

## Run 70: candidate construction audit
- Checking physical response signs, clone tensor differences, and exact local head interface.
- Will distinguish a preservation-only edit from the subsequent causal correction.
- Python runs: 70; forward passes: 4.

## Runs 72–73: corrected posterior-candidate frontier
- Run 72 identified the tuple layout but stopped because future_index59 contains (stay, hour) keys.
- Run 73 uses the retained boolean observed59 mask; candidate diagnostics are in diagnostics73.
- Scratch weights restored after all tests; no submitted edits.
- Python runs: 73; forward passes: 4.

## Run 75: reconstruction failure audit
- Run 74 stopped at an internal assertion; it did not validate or reject an edit.
- Recovering the exact core contract and retained candidate diagnostics before testing weights.
- Forward passes: 4; Python runs: 75.

## Run 76: stay-boundary diagnosis
- Run 74's assertion checks adjacent stay IDs, not numerical reconstruction accuracy.
- Inspecting candidate diagnostics and prior feature indexing; initial bundles require separate handling.
- Forward passes: 4; Python runs: 76.

## Run 77: boundary-safe core and prior-route audit
- Posterior-only candidates preserve local fit but do not recover identified dose slopes.
- Checking scratch weights, exact state reconstruction, and prior-head candidate interfaces.
- Forward passes: 4; Python runs: 77.

## Run 78: prior decoder preparation
- Boundary-safe original core agrees with saved states (out max error .00013).
- Original weights are restored. Inspecting reserved posterior features and prior capture before fitting a centered correction.
- Forward passes: 4; Python runs: 78.

## Run 79: units and prior sensitivity check
- Reserved posterior features encode centered dose innovations; candidate prior effects must be calibrated in physical rate units.
- Checking four-sensor signs, counterfactual fields, and retained dynamic targets before the first external candidate test.
- Forward passes: 4; Python runs: 79.

## Run 80: explicit carrier factual assay
- Built boundary-safe original/carrier prior datasets on 3935 bundles.
- Carrier local factual dNLL: 0.0000289; no submitted edits.
- All scratch weights restored. Forward passes: 4; Python runs: 80.

## Run 81: held-out prior bridge frontier
- Generated five multipliers for each drug using the exact action encoder and small posterior carrier.
- Fitted non-gradient, preservation-constrained prior dose bridges; held out stays divisible by five.
- Diagnostics retained in diagnostics81; weights remain unsubmitted.
- Forward passes: 4; Python runs: 81.

## Run 82: stronger dose-carrier frontier
- Small-carrier bridge failed held-out preservation/response trade-off; no candidate was submitted.
- Tested 10x, 100x, and 1000x carrier strengths with local factual and held-out bridge diagnostics.
- Results retained in carrier_results82/diagnostics82; scratch weights restored.
- Forward passes: 4; Python runs: 82.

## Run 88: unrestricted state-token assay
- Measured numerical response to all 192 z_in token coordinates.
- Tested explicit dose-innovation decoder calibration and 30-coordinate token compression.
- No submitted edits; forward passes remain 4.

## Run 90: aligned state-token candidate frontier
- Fixed factual reading-row mask alignment; tested compression-only and causal scales.
- Results retained in diagnostics90/candidates90; all scratch weights restored.
- No submitted edits; forward passes remain 4.

## Run 91: restart and token-route audit
- Run 90 preserved local factual fit, but insulin response remained weak and NE produced substantial glucose cross-effects.
- Recovering architecture, identification evidence, and candidate interfaces before choosing a drug-specific correction. No weights edited.
- Python runs: 91; forward passes: 4.

## Run 92: compact candidate recovery
- Architecture read successfully; shared latent-token edits can alter all heads.
- Run 91 output truncated before candidate interfaces; retrieving them compactly.
- No weights edited. Python runs: 92; forward passes: 4.

## Run 93: posterior clipping and unit audit
- Inspecting candidate construction, response targets, and exact local core before another response calibration.
- No weights edited. Python runs: 93; forward passes: 4.

## Run 94: factual reconstruction audit
- Drug response targets and signs agree with record identification.
- Local original NLL (~5.06) differs from the earlier external baseline (~1.63); auditing saved predictions and index contracts before trusting local preservation scores.
- No weights edited. Python runs: 94; forward passes: 4.

## Run 95: same-stay factual mismatch
- External original factual NLL on training-100 is 1.63805; local score on these stays is ~5.064. The mismatch is not explained by train/tuning.
- Auditing readout implementation and identical-reading alignment. Local edit scores are provisional until this passes.
- No edits. Python runs: 95; forward passes: 4.

## Runs 96–97: filtered-state mapping audit
- Run 96 stopped at an assertion: some reading rows were missing from the filtered state table. No predictions or weights changed.
- Rebuilt mapping directly from idx80, excluded missing entries, and completed exact-key comparison plus fresh original reconstruction.
- No edits. Python runs: 97; forward passes: 4.

## Run 99: restart after label audit
- Run 98 found 2250 incorrect old target bins; direct labels reduce but do not resolve the local/external NLL gap.
- Recovering architecture and identical-reading diagnostics before any weight edit.
- Python runs: 99; forward passes: 4; no submitted edits.

## Run 100: compact alignment recovery
- Architecture uses a recurrent state plus code-specific last-reading registers.
- Retrieving recent diagnostic objects and function names without the previous output truncation.
- Python runs: 100; forward passes: 4; no submitted edits.

## Run 101: head reconstruction audit
- Record-derived immediate effects remain approximately insulin -2 and NE +3, with persistence.
- Inspecting exact matched-reading errors and head/feature implementations before trusting local edit scores.
- Python runs: 101; forward passes: 4; no submitted edits.

## Run 102: timing and register audit
- Local errors are systematic, especially arterial MAP; next-time rounding and register construction are under examination.
- Python runs: 102; forward passes: 4; no submitted edits.

## Run 103: original-weight and worst-row audit
- Extreme errors on low readings are inconsistent with ordinary prior-sampling differences.
- Checking scratch weights, active hooks, exact registers, and predictions from external states.
- Python runs: 103; forward passes: 4; no submitted edits.

## Run 104: head convention experiment
- All scratch tensors are original and registers match raw records at the largest-error rows.
- Testing type/status indices and register age units against external factual predictions.
- Python runs: 104; forward passes: 4; no submitted edits.

## Run 105: last-numeric layout experiment
- Type/status and age-unit changes did not resolve the local/external prediction mismatch.
- Testing last-reading numeric field positions and head activations, without editing weights.
- Python runs: 105; forward passes: 4; no submitted edits.

## Run 106: exact reconstruction recovery
- Last-numeric layout changes reduced but did not eliminate the external mismatch.
- Re-reading the architecture and recovering local reconstruction interfaces before trusting any edit score.
- Python runs: 106; forward passes: 4; no submitted edits.

## Run 107: register interface inspection
- Architecture recovered; checking actual last-reading tensor dimensions and local head implementation.
- No submitted edits. Python runs: 107; forward passes: 4.

## Run 108: compact readout recovery
- last_numeric has 8 actual inputs versus 4 documented fields; previous output was truncated.
- Retrieving only readout bytecode and identification summary. No edits.
- Python runs: 108; forward passes: 4.

## Run 109: head-gap and alignment recovery
- Local readout shares the prior gap scaling; architecture specifies a distinct readout scaling.
- Checking exact matched-reading inputs before testing this candidate reconstruction fix.
- No edits. Python runs: 109; forward passes: 4.

## Run 110: explicit head-gap assay
- Best matched-reading convention: {'gap': 0.1, 'value_col': 6, 'flag_col': 3, 'activation': 'relu', 'nll': 2.025536060333252, 'mean_mae': 3.719050198109944, 'nll_mae': 0.722525797879013}
- Tested preprocessing only; no weights edited.
- Python runs: 110; forward passes: 4.

## Run 111: query-type and no-value audit
- Head gap scaling 1 versus .1 worsens reconstruction; rejected as the sole mismatch.
- Tested all 15 query types and inspected no-value mass; no model edits.
- Python runs: 111; forward passes: 4.

## Runs 112–113: fine-bin boundary assay
- Run 112 stopped at a tuple-key lookup error before scoring.
- Completed assay; best result: {'last_side': 'right', 'target_side': 'left', 'value_col': 6, 'activation': 'relu', 'nll': 2.0230093002319336, 'mean_mae': 3.722056214761734, 'nll_mae': 0.7211385927983249}
- No edits. Python runs: 113; forward passes: 4.

## Run 114: restart and evidence recovery
- Fine-bin conventions do not explain the factual reconstruction gap.
- Recovering architecture, early identification evidence, and available forward artifacts before selecting a repair.
- No edits. Python runs: 114; forward passes: 4.

## Run 115: compact recovery
- Early cross-fitted estimates support insulin glucose effects near -2 and NE MAP effects near +3 per rate unit.
- Recovering dynamic estimates and candidate interfaces; no edits.
- Python runs: 115; forward passes: 4.

## Run 116: targeted interface recovery
- Identified effects persist over several hours; repair must preserve dynamics and distinguish drugs.
- Recovering readout implementation and retained candidate diagnostics.
- No edits. Python runs: 116; forward passes: 4.

## Run 117: head-input and route inspection
- Retained token candidates do not yet deliver drug-specific target responses; none externally tested.
- Inspecting register preprocessing and saved mismatch examples before selecting the next route.
- No edits. Python runs: 117; forward passes: 4.

## Run 118: joint register-layout search
- Tested combined value/presence layouts for the actual eight-input last_numeric projection.
- Best full-probe result: {'value_col': 6, 'flags': (3,), 'activation': 'relu', 'objective': 0.8532955150320362, 'mean_mae': 3.7220562445640564, 'nll_mae': 0.7211386122008786}
- Preprocessing assay only; no weights edited. Python runs: 118; forward passes: 4.

## Run 119: empirical register-input reconstruction
- Joint discrete layouts failed; fit a 24-parameter preprocessing model to matched external outputs, holding all model weights fixed.
- Stay-held-out validation reported above; coefficients retained as B119.
- No edits. Python runs: 119; forward passes: 4.

## Run 120: state/latent readout convention assay
- Register-only reconstruction is inadequate; tested state, flow, gap, and latent conventions with original weights.
- Best result: {'out': 1, 'flow': 1, 'gap': 0.1, 'latent': 'zero', 'layout': (6, 3), 'activation': 'relu', 'objective': 0.842706940385124, 'mean_mae': 3.7176732511520387, 'nll_mae': 0.7126167141485338}
- No edits. Python runs: 120; forward passes: 4.

## Run 121: external-validation reset
- Local readout conventions remain inconsistent with official predictions; local factual scores are not trusted.
- Recovering architecture, record identification, and repair interfaces before forward validation.
- No submitted edits. Python runs: 121; forward passes: 4.

## Run 122: dynamic-target recovery
- Cross-fitted immediate estimates support insulin glucose effects near -2 and NE MAP effects near +3 per rate unit.
- Recovering persistence and retained repair interfaces; no edits submitted.
- Python runs: 122; forward passes: 4.

## Run 123: centered correction machinery
- Dynamic targets persist across hours; original dose responses are weak, especially for NE.
- Inspecting existing centered action/latent candidates to avoid restarting reconstruction.
- No submitted edits. Python runs: 123; forward passes: 4.

## Run 124: token-candidate inspection
- Exact recurrent reconstruction is available, but local value readout remains untrusted.
- Recovering candidate tensors and diagnostics for an official factual test rather than more readout searches.
- No submitted edits. Python runs: 124; forward passes: 4.

## Run 125: moderate token candidate submitted for testing
- Saved candidates90[1.0], six posterior/token tensors, as current edits.
- Local diagnostics suggest small factual changes but weak insulin response and NE-to-glucose cross-effects; these are provisional.
- Next: official tuning factual and paired rollouts; reject if fit fails or routing is inadequate.
- Python runs: 125; forward passes: 4.

## Run 126: official rejection and drug-separated calibration
- Forward 5 factual: original 1.63155, candidate 1.63154; preservation passed on tuning-100.
- Forward 6 rollout candidate: insulin x2 -1.242, half +0.495; NE x2 -0.666, half -0.395.
- Candidate response is worse than original and reverses NE doubling; restored all original weights.
- Decomposing token correction by drug for independent official calibration; local response signs are not reliable.
- Python runs: 126; forward passes: 6.

## Run 127: architecture and causal-target recovery
- Previous candidate passed factual fit but worsened intervention responses; originals remain restored.
- Reading architecture and retained function interfaces before designing a drug-separated repair.
- Python runs: 127; forward passes: 6; no submitted edits.

## Run 128: candidate construction and baseline rollout audit
- Record impulse estimates: insulin/glucose -2.073 at one hour, NE/MAP +2.952; glucose persists longer.
- Inspecting the latent projection candidate and official baseline rollout before another edit.
- Python runs: 128; forward passes: 6; no submitted edits.

## Run 129: latent edit recipe and exposure scale
- Official original tuning rollout: insulin x2 -1.518, half +0.791; NE x2 +0.169, half -0.038.
- Recovering latent candidate construction and rate scale to connect causal impulse estimates to rollout targets.
- Python runs: 129; forward passes: 6; no submitted edits.

## Run 130: distinguish compression from intervention channels
- Candidate baseline rotates/compresses the original latent token; scale-dependent changes must be measured relative to that baseline.
- Auditing the two correction channels and their innovation targets before independent calibration.
- Python runs: 130; forward passes: 6; no submitted edits.

## Run 131: audit physical-dose decoding before rollout calibration
- Correction channels occupy latent coordinates 30 and 31; their token directions have norms 3.867 and 5.438.
- Posterior first-layer changes are large, so dose monotonicity and drug separation require direct verification.
- Python runs: 131; forward passes: 6; no submitted edits.

## Run 132: physical-dose channel and convolution-target tests
- Directly tested latent correction channels under factors 0, 0.5, 1.5, 2, and 3.
- Computed approximate tuning rollout targets by convolving recorded doses with empirical drug impulse responses.
- Targets are identification-based estimates, not observed counterfactual truth; tail extrapolation is reported separately.
- Python runs: 132; forward passes: 6; no submitted edits.

## Run 133: physical units and official factual perturbation audit
- Approximate tuning targets: insulin x2 -10.37 to -11.39, half +5.18 to +5.70; NE x2 +4.21 to +4.25, half -2.10 to -2.13.
- Auditing raw infusion rates against retained decoder targets and measuring official factual mean changes of the rejected candidate.
- Python runs: 133; forward passes: 6; no submitted edits.

## Run 135: recover recurrent calibration interface
- Decoder intervention differences mostly track physical-dose changes.
- Prior-feature assay failed because context was not captured.
- Recovering interfaces before official drug-separated calibration.
- No edits. Python runs: 135; forward passes: 6.

## Run 136: targeted latent/recurrent interface recovery
- Architecture separates posterior token effects from prior-head predictions.
- Inspecting retained calibration recipes and early identification evidence.
- No edits. Python runs: 136; forward passes: 6.

## Run 137: inspect separable correction channels
- Record estimates support drug-specific effects and persistence; remain observationally identified.
- Inspecting baseline compression and channel directions before official calibration.
- No edits. Python runs: 137; forward passes: 6.

## Run 138: submit zero-correction baseline for official assay
- Candidate scale changes only two reserved drug-token columns.
- Submitted candidates90[0.0]; baseline reconstruction needs independent rollout validation.
- Python runs: 138; forward passes: 6.
- Calibration stays: [2020000000, 2020000016, 2020000032, 2020000048, 2020000064, 2020000080, 2020000096, 2020000001, 2020000017, 2020000033, 2020000049, 2020000065, 2020000081, 2020000097, 2020000002, 2020000018, 2020000034, 2020000050, 2020000066, 2020000082, 2020000098, 2020000003, 2020000019, 2020000035, 2020000051, 2020000067, 2020000083, 2020000099, 2020000004, 2020000036, 2020000052, 2020000068, 2020000084, 2020000005, 2020000021, 2020000037, 2020000053, 2020000069, 2020000085, 2020000006, 2020000022, 2020000038, 2020000054, 2020000070, 2020000086, 2020000007, 2020000023, 2020000039, 2020000055, 2020000071, 2020000087, 2020000008, 2020000024, 2020000040, 2020000056, 2020000072, 2020000088, 2020000009, 2020000025, 2020000041, 2020000057, 2020000073, 2020000089, 2020000010, 2020000026, 2020000042, 2020000058, 2020000074, 2020000090, 2020000011, 2020000027, 2020000043, 2020000059, 2020000075, 2020000091, 2020000012, 2020000028, 2020000044, 2020000060, 2020000076, 2020000092, 2020000013, 2020000029, 2020000045, 2020000061, 2020000077, 2020000093, 2020000014, 2020000030, 2020000046, 2020000062, 2020000078, 2020000094, 2020000015, 2020000031, 2020000047, 2020000063, 2020000079, 2020000095]

## Forward 7 / Run 139: baseline validated; reversed-channel assay
- Zero-correction rollout closely matches original: insulin x2 -2.002 vs -2.016, half +1.058 vs +1.051; NE x2 +0.316 vs +0.309, half -0.077 vs -0.070.
- Submitted independently scalable channels at insulin -4 and NE -4, motivated by the wrong-direction response of the prior +1 candidate.
- Next official factual test checks the 0.005-nat constraint, followed by paired rollout.
- Python runs: 139; forward passes: 7.

## Run 140: audit after reversed-channel rollout
- Forward 9: insulin x2 -2.750, half +1.370; NE x2 +0.210, half +0.095.
- Reversed channels modestly strengthen insulin but fail NE; no further edits in this run.
- Recovering architecture, official factual constraint, and retained calibration machinery.
- Python runs: 140; forward passes: 9.

## Run 141: factual margin and identification recovery
- Forward 8 value NLL delta: +0.002286 nats/reading on tuning-99; within constraint.
- Current rollout remains far short of identification-based drug targets, especially NE.
- Recovering empirical evidence and correction-channel construction before redesign.
- Python runs: 141; forward passes: 9.

## Run 142: reconstruct channel recipe
- Identification evidence supports insulin/glucose and NE/MAP effects with persistence.
- Existing token directions were calibrated with unvalidated local value readouts.
- Inspecting candidate construction to retain dose centering but replace response routing.
- Python runs: 142; forward passes: 9; edits unchanged.

## Run 143 / Forward 10: paired authoritative states
- Loaded original/edited tuning-99 states and aligned factual readings to strict pre-bundle states.
- Rebuilding readout calibration from official predictions, without post-reading leakage.
- Python runs: 143; forward passes: 10; edits unchanged.

## Run 144: authoritative readout-surrogate validation
- Fit stay-held-out smooth ridge readouts, with last-value splines and state features.
- Validation includes original factual means and current-edit paired mean changes.
- Results retained as models144/diagnostics144; no weight changes.
- Python runs: 144; forward passes: 10.

## Run 145: paired-response surrogate assay
- Original-only readouts failed MAP edit-delta validation; not used for routing.
- Added paired official state/prediction changes to ridge fits, retaining stay-held-out validation.
- Results retained as models145/diagnostics145; edits unchanged.
- Python runs: 145; forward passes: 10.

## Run 146: prepare validated-surrogate token routing
- Paired held-out delta R2: finger .787, lab .848, arterial .809, cuff .723.
- Recovering exact local update and counterfactual feature interfaces for finite-difference linear editing.
- No new saved weights; Python runs: 146; forward passes: 10.

## Run 147: recover validated routing interfaces
- Current candidate has factual margin but underestimates both drug effects.
- Recovering exact local update, latent candidate, and paired readout interfaces to replace unvalidated token directions; no edits changed.
- Python runs: 147; forward passes: 10.

## Run 148: compact interface recovery
- Architecture confirms posterior drug channels reach the recurrent state through z_in.
- Run 147 output truncated; now capturing actual local-update inputs and paired readout structure without a full disassembly.
- No saved weight changes. Python runs: 148; forward passes: 10.

## Run 149: targeted update/feature recovery
- Paired readouts use state features with last-reading interactions.
- Previous diagnostic output again truncated; printing only missing update context, candidate recipe, and feature-block checks.
- No saved changes. Python runs: 149; forward passes: 10.

## Run 150: local reconstruction and feature-layout checks
- Exact captured update inputs are available; checking reconstruction against official original states.
- Reserved drug innovation channels are retained; feature ordering must be established before computing new response directions.
- No saved edits changed. Python runs: 150; forward passes: 10.

## Run 151: reconstruct readouts and physical-dose channels
- Local original output reconstruction RMS is 1.19e-5.
- Undoing readout standardization, verifying interactions, and capturing training drug contexts with strictly prior reading registers.
- No saved edits changed. Python runs: 151; forward passes: 10.

## Run 152: recover complete routing and identification interfaces
- Local original-state reconstruction is numerically exact; retained dose channels are approximately physical and drug-specific.
- Inspecting architecture, schema, existing helpers and early identification evidence before a closed-form routing edit.
- No weights changed. Python runs: 152; forward passes: 10.

## Run 153: compact finite-difference calibration interfaces
- Architecture routes posterior dose innovations through z_in and the recurrent state.
- Restricting inspection to exact local update, validated paired readouts, and retained identification estimates.
- No weights changed. Python runs: 153; forward passes: 10.

## Runs 154-155: repair derivative interface and assay existing directions
- Run 154 stopped before computation: retained readoutweights151 already contains separate 192-dimensional view/interaction coefficients.
- Corrected that interface and measured exact-core finite-difference responses; recovering physical action contrast helper.
- No saved weights changed. Python runs: 155; forward passes: 10.

## Runs 156-157: drug/bundle alignment and exact-core Jacobian
- Run 156 stopped before differentiation because joint-drug bundles require drug-specific alignment keys.
- Corrected alignment by drug and bundle; computed central finite-difference token/readout sensitivities.
- Record cross-fitted immediate effects: finger -1.935, lab -2.015 per insulin unit; arterial +3.053, cuff +2.528 per NE unit.
- No saved weights changed. Python runs: 157; forward passes: 10.

## Run 158: closed-form drug-specific token routing
- Original local responses understate identified effects: insulin/finger about -0.468; NE/arterial about +0.070 per rate unit.
- Fit ridge response directions using a numerical exact-core Jacobian, drug-specific physical-dose contrasts, and stay-held-out validation.
- Tested nonlinear local responses and factual mean perturbations of the best regularization settings.
- Retained candidate158, alpha=1e-06; not saved. Python runs: 158; forward passes: 10.

## Run 159: reject nonlinear-saturating routing candidate
- Candidate158 was not saved: nonlinear dose responses remain weak despite very large token directions.
- Linear/surrogate fit is insufficient outside the validated edit regime; auditing token scale and actual value-head reconstruction.
- Current saved edits remain candidate139. Python runs: 159; forward passes: 10.


## Run 160: exact-head redesign
- Rejected candidate158 because token magnitude and nonlinear saturation invalidate its linear calibration.
- Recovering the actual prediction path and record-based dynamic effects before further editing.
- Saved edits remain candidate139; Python runs: 160; forward passes: 10.


## Run 161: smaller channel plus explicit readout route
- Architecture and retained identification suggest testing a small centered-dose state channel with a separately amplified value-head route.
- Recovering actual-head reconstruction and local update helpers; no weights changed.
- Python runs: 161; forward passes: 10.


## Run 162: audit readout validation before redesign
- Earlier absolute-head reconstruction was not validated; do not treat its NLL as authoritative.
- Inspecting compact helper interfaces for a register-independent dose decoder and explicit value-head correction.
- No weights changed. Python runs: 162; forward passes: 10.


## Run 163: dose-channel separability assay
- Testing whether paired edited queries contain centered-dose information in directions with little original-query variation.
- Such separation would allow head amplification without large recurrent tokens; no weights changed.
- Python runs: 163; forward passes: 10.


## Runs 164–166: complete register-independent decoder assay
- Recovered both drug contrast sets, aligned by bundle and drug; fixed NumPy/Torch action interface.
- Decoders trained on exact local contrasts with a penalty on original-query noise; held-out diagnostics printed.
- No weights changed. Python runs: 166; forward passes: 10.

## Run 167: reject noisy head-only dose decoding
- Held-out register-independent decoders recover incomplete dose changes; amplification incurs substantial original-query noise.
- Reviewing exact architecture and retained record-based effects before choosing a safer route. Saved edits unchanged.
- Python runs: 167; forward passes: 10.

## Run 168: compact architecture and evidence recovery
- Architecture confirms a nonlinear shared state/readout path; run 167 was truncated before retained variables and evidence.
- Retrieving only recent interfaces, relevant tensor shapes, and record-based identification notes. No weights changed.
- Python runs: 168; forward passes: 10.

## Run 169: capped identification/interface extraction
- Recent exact-update helpers are retained. Avoiding tensor-valued function defaults, which caused run 168 output overflow.
- Extracting effects, persistence, and exact local interfaces before any new edit. Saved edits unchanged.
- Python runs: 169; forward passes: 10.

## Run 170: reconsider regularized value-head correction
- Immediate record estimates: insulin about -2 per rate unit; NE arterial MAP about +3.05; glucose persistence exceeds MAP persistence.
- High-penalty decoder noise may be tolerable after physical scaling. Preparing a no-gradient, factual-penalized correction using existing value-head features rather than large recurrent tokens.
- Saved edits unchanged. Python runs: 170; forward passes: 10.

## Run 171: prepare factual-penalized head-feature assay
- High-penalty dose decoding, after slope rescaling, implies roughly 0.4-0.5 glucose-unit and 0.3-0.4 MAP-unit factual noise, not an automatic rejection under the NLL budget.
- Recovering head inputs and exact drug/bundle alignment to test existing-hidden-feature corrections. Saved edits unchanged.
- Python runs: 171; forward passes: 10.

## Run 174: restart and normalization recovery
- Run 173 stopped at a normalization lookup; no new edit was evaluated.
- Recovering architecture, physical units, retained identification evidence, and exact readout interfaces before the next candidate.
- Python runs: 174; forward passes: 10. Saved edits unchanged.

## Runs 175–176: normalization lookup corrected
- num_spec0 is a dictionary keyed by (event, attribute); run 175's disassembly attempt failed before candidate diagnostics.
- Recovering compact interfaces and official forward-10 results.
- Python runs: 176; forward passes: 10. Saved weights unchanged.

## Run 177: recover exact hidden-head inputs
- Official saved-candidate factual delta is +0.002286; rollout effects remain insulin x2 -2.750 and NE x2 +0.210, too small for record estimates.
- Recovering retained head reconstruction for a small final-layer correction with an explicit factual penalty.
- Python runs: 177; forward passes: 10. No weights changed.

## Run 178: paired-head feature capture
- Repeated run-177 output supplied no new result; proceeding with capture.
- Absolute local-head NLL remains non-authoritative; candidate acceptance will use official forward passes.
- Capturing hidden head features and recovering physical action contrasts.
- Python runs: 178; forward passes: 10. No saved changes.

## Run 179: build paired and factual output-layer features
- Captured original/doubled-dose hidden features and probabilities for all four sensors, plus 1,600 factual contexts.
- Original local-core reconstruction checked against official states.
- Preparing a no-gradient rank-one bin-tilt edit, with factual Fisher penalty and stay-held-out dose-response validation.
- Python runs: 179; forward passes: 10. Saved edits unchanged.

## Run 180: closed-form output-layer correction
- Fitted missing record-based immediate effects jointly for all four sensors; cross-drug effects constrained toward zero.
- Used original-head factual Fisher penalty, ridge regularization, and stay-held-out nonlinear head replay, not model gradients.
- Selected diagnostics: {'lambda': 0.03, 'kl': 0.0007810858545918381, 'rms': {'glucose_finger': 0.3440168093377411, 'glucose_lab': 0.38350028887760224, 'map_arterial': 0.23346797245831993, 'map_cuff': 0.3205948462785093}, 'slopes': {'glucose_finger': -0.1712618083048881, 'glucose_lab': -0.14250014693584145, 'map_arterial': 0.13919880578360694, 'map_cuff': 0.17244300529981044}, 'rmse': 0.04566138598871352}
- candidate180 retained, not saved; absolute local-head NLL remains unvalidated.
- Python runs: 180; forward passes: 10.

## Run 181: restart and exact-path audit
- Run 180's low-distortion head correction barely changed drug sensitivity; it remains unsaved.
- Recovering architecture, identification evidence, saved candidate139, and recent interfaces before another edit.
- Python runs: 181; forward passes: 10. No weights changed.

## Run 182: bounded recovery
- Architecture recovered; run 181 output was truncated before useful retained interfaces.
- Retrieving bounded identification, candidate139, and recent object summaries.
- Python runs: 182; forward passes: 10. Saved weights unchanged.

## Run 183: scaling and persistence audit
- Record identification and saved-edit factual margin recovered.
- Inspecting retained head-fit arrays and dynamic estimates to distinguish scaling error from routing failure.
- Python runs: 183; forward passes: 10. No weights changed.

## Run 184: head signal and reserved-channel audit
- Dynamic record estimates show multi-hour persistence, requiring a recurrent or feedback-compatible repair.
- Checking head-fit units, helper dependencies, and saved latent-channel construction.
- Python runs: 184; forward passes: 10. No saved changes.

## Run 185: reserved physical-dose channel assay
- Original-head correction has little counterfactual signal; testing reserved posterior dose contrasts directly.
- Scratch weights and monkeypatch restored after replay; saved edits unchanged.
- Python runs: 185; forward passes: 10.

## Runs 186–187: actual-head derivative interface corrected
- Run 186 stopped before fitting because local heads retain 16 prior draws per context.
- Averaged prior draws and repeated finite differences; retained factual-penalized directions.
- Scratch weights restored; saved edits unchanged. Python runs: 187; forward passes: 10.

## Run 188: nonlinear held-out token routing
- Replayed true local value heads under all finite-difference candidates and current saved139 directions.
- Best low-distortion diagnostic: {'label': 'ridge_1e-05', 'kl': 0.00042278443367740784, 'rms': {'glucose_finger': np.float64(0.2697626139125298), 'glucose_lab': np.float64(0.2701111808990397), 'map_arterial': np.float64(0.18902725564208214), 'map_cuff': np.float64(0.2502358226235345)}, 'slopes': {'glucose_finger': -0.19590929766060625, 'glucose_lab': -0.16433030540049706, 'map_arterial': 0.9332077099836112, 'map_cuff': 1.251882103801596}, 'cross': {'glucose_finger': 0.2896998061828172, 'glucose_lab': 1.1037621180799433, 'map_arterial': 0.024963782716179744, 'map_cuff': 0.027810183630891726}, 'score': 0.15968022214739647}
- No new weights saved. Python runs: 188; forward passes: 10.


## Run 189: restart and bounded recovery
- Run 188 improved local norepinephrine routing but not insulin; local KL is not an official factual-fit result.
- Recovering exact architecture, record schema, and retained routing interfaces before selecting another edit.
- No weights changed. Python runs: 189; forward passes: 10.


## Run 190: recover evidence and exact routing construction
- Architecture places dose effects behind nonlinear latent/state routing; weak local insulin response is not evidence of a weak true effect.
- Recovering identification estimates, candidate construction, and exact replay helpers.
- No weights changed. Python runs: 190; forward passes: 10.


## Run 191: compact interface recovery
- Run 190 output truncated; record evidence still supports insulin about -2 and NE about +3 per rate unit.
- Retrieving only routing helper dependencies and recent fit arrays to avoid another output overflow.
- No weights changed. Python runs: 191; forward passes: 10.


## Run 192: physical channel versus downstream bottleneck
- Inspecting reserved-dose channel ranges, candidate construction, and output-direction norms.
- Distinguishing dose-encoding failure from nonlinear downstream saturation before official testing.
- No weights changed. Python runs: 192; forward passes: 10.


## Run 193: submit smaller statistically calibrated routing candidate
- Reserved physical-dose channels respond substantially to doubled doses; downstream routing is the bottleneck.
- Replaced only the zero-candidate z_in reserved columns 30:32 with best188's fitted directions; retained baseline-compression posterior weights.
- Saved candidate193; previous saved candidate139 retained in backup193.
- Awaiting official factual and rollout validation. Python runs: 193; forward passes: 10.


## Forwards 11–12 / Run 194: reject reconstructed-head calibration
- Candidate193 official factual delta -0.00045 nats/reading.
- Official rollout: insulin x2 -1.183, half +0.520; NE x2 -0.049, half -0.365. Worse than baseline despite promising local-head slopes.
- Therefore local reconstructed value heads are not a reliable dose-response surrogate.
- Starting a statistical readout calibrated directly to official original predictions and states; no new weights changed.
- Python runs: 194; forward passes: 12.

## Run 196: official-head audit
- Recovered record-based immediate and dynamic effects; original rollout is substantially under-responsive.
- Candidate193 local diagnostics were refuted by official rollout. Auditing exact head reconstruction before another calibrated edit.
- Python runs: 196; forward passes: 12. Saved edits unchanged.

## Run 197: inspect authoritative surrogate
- Zero-channel compressed baseline matched original official rollout. Drug directions, rather than baseline compression, need replacement.
- Inspecting retained official-prediction surrogate and physical-dose-channel interfaces.
- Python runs: 197; forward passes: 12. Saved edits unchanged.

## Run 198: recover paired-surrogate routing results
- Original-only surrogate failed edit-delta validation; paired surrogate achieved held-out delta R2 of about .72–.85.
- Recovering its retained routing candidates before further official validation.
- Python runs: 198; forward passes: 12. Saved edits unchanged.

## Run 199: official sign-reversal experiment
- Paired-surrogate candidate158 required large directions and saturated; it was never submitted.
- Candidate193's small directions moved official doubled-dose responses in the wrong direction, despite a factual delta of -0.00045.
- Submitted -2 times those two added z_in columns, with the independently validated zero-channel baseline unchanged.
- Hypothesis: reversal strengthens insulin lowering and NE raising; official factual and rollout tests will decide.
- Previous saved tensors retained in backup199. Python runs: 199; forward passes: 12.

## Run 200: restart after sign-reversal validation
- Forward 14: reversed routing modestly improved insulin (x2 -2.682, half +1.339) but NE remained wrong (x2 -0.069, half +0.369).
- Recovering architecture and record evidence; a shared sign change is not an adequate repair. No weights changed.
- Python runs: 200; forward passes: 14.

## Run 201: recover dynamics and current factual margin
- Architecture and schema recovered. Earlier cross-fitted immediate effects were approximately insulin -2 and NE +3 per rate unit.
- Retrieving retained interfaces and official forward-13 factual loss before testing drug-specific directions. No weights changed.
- Python runs: 201; forward passes: 14.

## Run 202: bounded routing-interface audit
- Current factual delta is +0.002308 nats/reading, inside the +0.005 limit but not enough evidence to accept incorrect drug responses.
- Inspecting reserved-channel candidates, core/head helpers, and record-derived dynamics. No weights changed.
- Python runs: 202; forward passes: 14.

## Run 203: bounded head-source recovery
- Run 202 confirmed retained physical channels and six-tensor candidate interfaces; output overflow hid the head details.
- Avoiding tensor-valued function defaults and retrieving focused head/token dependencies. No weights changed.
- Python runs: 203; forward passes: 14.

## Run 204: audit dose encoding independently of invalid heads
- Record impulse responses persist for several hours; one-step head slopes alone cannot establish rollout repair.
- Earlier local-head reconstructions were not authoritative. Recovering exact dose-channel and probe construction.
- Python runs: 204; forward passes: 14. No weights changed.

## Run 205: directly measure reserved-dose channels
- Measuring factual, doubled-dose, and empty-action posterior channels using the actual saved posterior weights, without a value-head surrogate. No weights changed.
- Python runs: 205; forward passes: 14.

## Run 207: recover authoritative replay interface
- Run 206 measured nonzero reserved-channel variation, then failed on an empty selection; this was a diagnostic indexing failure.
- Reading architecture and schema and locating retained replay functions before further edits.
- No weights changed. Python runs: 207; forward passes: 14.

## Run 208: focused dose-channel recovery
- Architecture confirms parallel direct-action and posterior-latent routes; a common latent sign reversal is not a drug-specific causal calibration.
- Recovering exact physical-channel construction and candidate interfaces with bounded output.
- No weights changed. Python runs: 208; forward passes: 14.

## Run 209: official test-history comparison
- Comparing all official rollout/factual results before selecting new routing directions; local-head diagnostics alone were refuted.
- No weights changed. Python runs: 209; forward passes: 14.

## Run 210: assess guarded physical-dose routing
- Repeated run-209 output supplied no new validation. Official tested edits remain far from record-derived targets.
- Inspecting reserved neurons and replay interfaces to assess stronger routing guarded against factual dose variation.
- No weights changed. Python runs: 210; forward passes: 14.

## Run 211: distinguish reserved neurons and bundle-input errors
- Run 210 output truncated; visible dose channels are distributed posterior readouts, not simple two-neuron gates.
- Checking reserved input rows and official bundle flags before altering their construction.
- No weights changed. Python runs: 211; forward passes: 14.

## Run 212: verify empty-action convention against official states
- Local empty-action vectors have norm 11.67 despite architecture specifying zero; testing this as a replay mismatch, not assuming it is causal evidence.
- Comparing original posterior means against official state dumps under both conventions.
- No weights changed. Python runs: 212; forward passes: 14.

## Run 213: reject zero-empty-action change; audit downstream surrogate
- Zeroing empty actions worsened original posterior RMS (0.105 to 0.112) and created large false reserved-channel values; rejected.
- Drug-bundle posterior RMS is 1.57e-5 under existing encoding, supporting its use for direct drug-bundle tests.
- Auditing official paired-state readout calibration before another routing edit. No weights changed.
- Python runs: 213; forward passes: 14.

## Run 214: recover authoritative architecture and targets
- Drug-bundle posterior encoding is validated; prior local value-head calibration is not.
- Recovering architecture, official results, and record identification evidence before further edits.
- No weights changed. Python runs: 214; forward passes: 14.

## Run 215: compact recovery after truncated output
- Architecture confirms nonlinear, parallel action/posterior routing.
- Recovering only official scalar results, causal evidence, and head implementation metadata.
- No weights changed. Python runs: 215; forward passes: 14.

## Run 216: quantify targets and untested routing candidate
- Official current shifts: insulin -2.682/+1.339; NE -0.069/+0.369.
- Training one-hour effects are approximately -2 and +3 per rate unit; persistence matters for rollout targets.
- Inspecting retained candidate158 and channel scales before selecting a test.
- No weights changed. Python runs: 216; forward passes: 14.

## Run 217: quantify cumulative targets and channel selectivity
- Computing six-lag record-identified rollout targets on the official tuning sample.
- Measuring factual versus doubled-dose channel activity and injection sizes for retained candidates.
- No weights changed. Python runs: 217; forward passes: 14.

## Run 218: submit untested paired-state routing candidate
- Six-lag record-identified tuning targets: insulin x2 -12.3739, half +6.1870; NE x2 +4.3837, half -2.1918. These omit any effects beyond lag six.
- Existing routing remains much too weak. Candidate158 uses paired official-state readout calibration rather than the refuted reconstructed value head.
- Submitted its six tensors; backup218 retains the previous edit.
- Large factual injections make preservation uncertain; official factual test comes first.
- Python runs: 218; forward passes: 14.

## Run 219: recover after candidate158 factual validation
- Forward 15 factual delta: +0.00002 nats/reading on 100 tuning stays, comfortably within the limit.
- Candidate158 remains submitted; its dose response still requires official rollout validation.
- Recovering architecture, schema, and tuning IDs before the next forward request.
- Python runs: 219; forward passes: 15.

## Run 220: candidate158 rollout refuted
- Forward 16: edited insulin x2/half -1.823/+1.007; NE x2/half -0.004/+0.009.
- Factual delta remains +0.00002, but paired-state calibration did not repair causal rollout.
- Inspecting saved physical-dose channels and downstream routing before another edit.
- Python runs: 220; forward passes: 16.

## Run 221: official latent-channel audit
- Forward 17 supplied paired factual states for candidate158.
- Measuring reserved posterior channels, latent-token injection, and recurrent-state differences.
- No weights changed. Python runs: 221; forward passes: 17.

## Run 222: active channels, weak downstream routing
- Official candidate158 channels are nonzero; median injection norms ~23-24.
- Output-state changes are much smaller, often nearly zero, despite sizable slot-state changes.
- This rejects an inactive-channel explanation and motivates a different downstream edit boundary.
- Recovering previous head experiments and targeted routing matrices; no weights changed.
- Python runs: 222; forward passes: 17.

## Run 223: inspect normalization and recurrent injection boundary
- Large latent-token injections did not yield the needed official response.
- Checking exact out normalization and interfaces for a low-rank core.b_in correction.
- No weights changed. Python runs: 223; forward passes: 17.

## Run 225: authoritative replay recovery
- Run 224 failed through tuple indexing in the local diagnostic; no weights changed.
- Current six-tensor candidate preserves factual fit but fails causal rollout.
- Recovering exact architecture and replay definitions before choosing a different edit boundary.
- Python runs: 225; forward passes: 17.

## Run 226: focused replay definitions
- Architecture recovered; prior output truncation hid diagnostics.
- Inspecting the core helper and argument types only, to resolve tuple indexing.
- No weights changed. Python runs: 226; forward passes: 17.

## Run 227: compact core and head recovery
- Confirmed retained replay context contains correctly shaped tensors.
- Recovering compact instructions for token substitution, recurrent update and value readout.
- No weights changed. Python runs: 227; forward passes: 17.

## Run 228: explicit prelude/recurrent replay
- Built tensor-only replay exposing E and the pre-coda recurrent state.
- Restored an original-weight local linear helper; submitted weights unchanged.
- Replay must match official original states before it guides core.b_in edits.
- Python runs: 228; forward passes: 17.

## Run 229: validate local core against official states
- Record-based immediate and multi-hour effects recovered; no medical assumptions used.
- Inspecting official state files and matching captured contexts before any core edit.
- No weights changed. Python runs: 229; forward passes: 17.

## Run 230: dose sensitivity at the prelude boundary
- Comparing factual versus doubled-dose E under original and scaled latent routing.
- Testing whether a low-rank core.b_in correction can use dose-sensitive features with low factual variance.
- No submitted weights changed. Python runs: 230; forward passes: 17.

## Run 231: prepare constrained recurrent correction
- Core replay validated: output RMS 1.38e-5; h RMS .00122; posterior RMS 1.57e-5.
- Factor-one action encoding is exact. Dose changes reach E but are attenuated in out.
- Recovering generic replay and head reference to constrain factual side effects.
- No weights changed. Python runs: 231; forward passes: 17.

## Run 232: recover exact edit boundary
- Original drug-bundle replay is validated, but current latent routing remains causally weak.
- Recovering architecture and helper names before a direct recurrent/readout correction.
- No weights changed. Python runs: 232; forward passes: 17.

## Run 233: inspect validated helper interfaces
- core.b_in is an available downstream dose-routing boundary; factual selectivity remains unproven.
- Recovering helper dependencies and record-identification evidence.
- No weights changed. Python runs: 233; forward passes: 17.

## Runs 234-235: explicit replay capture
- Run 234 omitted the helper's required action argument and stopped; no weights changed.
- Run 235 supplies recorded actions and captures core/readout internals.
- Python runs: 235; forward passes: 17.

## Run 236: test recurrent-feature selectivity
- Working replay exposes E and the head inputs.
- Quantifying minimum-factual-energy doubled-dose features before amplification.
- No weights changed. Python runs: 236; forward passes: 17.

## Run 237: physical feature fits and head validation
- E has potentially useful factual-quiet directions, but held-out factual effects are not tested.
- Checking original local means against official factual predictions and fitting actual rate contrasts.
- No weights changed. Python runs: 237; forward passes: 17.

## Runs 237-238: repair diagnostic joins
- Run 237's zero doses and empty head comparison were failed joins, not scientific results.
- Matching records across available splits with rounded bundle timestamps.
- No weights changed. Python runs: 238; forward passes: 17.

## Run 239: re-establish trustworthy edit interfaces
- Corrected joins validate dose encoding and reveal factual-quiet E features.
- Local value predictions remain unvalidated; do not use them to calibrate an edit yet.
- Recovering authoritative architecture and retained helper inventory in one pass.
- Submitted weights unchanged. Python runs: 239; forward passes: 17.

## Run 240: compact interface recovery
- Architecture confirms prelude-to-recurrent-to-coda attenuation of action signals.
- Previous inventory was truncated by tensor-valued function defaults.
- Listing parameter names and recent replay objects without defaults.
- No weights changed. Python runs: 240; forward passes: 17.

## Run 241: inspect correction and readout boundaries
- Existing core replay accepts an explicit correction; use it rather than rebuilding the model.
- Recovering only its injection semantics, head evaluator, and physical-rate feature fits.
- Aim: a small low-rank recurrent correction, validated by official factual and rollout passes.
- No weights changed. Python runs: 241; forward passes: 17.

## Runs 242-243: all-bundle constrained rate features
- Run 242 established core.b_in correction = delta * linear(E, correction), before coda.
- Local original doubled-dose shifts: insulin glucose -0.501; NE MAP +0.093.
- Run 242's final metadata loop failed after useful results; no weights changed.
- Captured factual E across all bundle types and fitted joint physical-rate contrasts.
- No weights changed. Python runs: 243; forward passes: 17.

## Run 244: numerical readout routing
- All-bundle constraints reduce rate-contrast slopes to ~0.29 insulin/~0.40 NE at penalty 1.
- Factual feature RMS is ~0.215 insulin/~0.087 NE; amplification requires official validation.
- Fitting pre-coda glucose/MAP directions by finite-difference response regression, without autograd.
- No weights changed. Python runs: 244; forward passes: 17.

## Run 245: multiplier-general low-rank recurrent candidate
- Numerical routing works locally; highly precise routes have large norms.
- Chose more regularised readout directions and fitted rate features across x0/.5/1.5/2/3.
- Computed a rank-two core.b_in candidate calibrated to record-estimated one-hour effects.
- Measuring exact one-step factual and dose responses before submission.
- No weights changed. Python runs: 245; forward passes: 17.

## Runs 247-249: metadata recovery
- Runs 247 and 248 stopped on inventory type errors, before any weight changes.
- Using explicit Tensor/ndarray branches to recover the candidate and replay functions.
- Python runs: 249; forward passes: 17.

## Run 250: recover assembled recurrent correction
- Candidate rate-feature and readout fits are retained; recovering assembled matrix and exact replay interface.
- No weights changed. Python runs: 250; forward passes: 17.

## Run 251: submit half-strength recurrent candidate
- Replaced the causally ineffective six-tensor latent candidate with rank-two core.b_in.weight correction.
- Strength 0.5; original latent weights restored. Previous candidate archived in /work/pre_recurrent_candidate.npz.
- Local factual shifts are modest, but official factual NLL and multi-hour rollout are not yet validated.
- Python runs: 251; forward passes: 17.

## Run 252: recover validation and identification evidence
- Forward 19, half-strength rank-two core.b_in edit: insulin x2 -5.961, half +2.696; NE x2 +1.967, half -0.192. Original: -2.016, +1.051, +0.309, -0.070.
- Responses strengthened but NE asymmetry remains; effect accuracy and factual eligibility must be checked against recovered evidence.
- Reading architecture, record schema, identification notes and forward 18 factual outputs. No weights changed.
- Python runs: 252; forward passes: 19.

## Run 253: compact factual and candidate recovery
- Recovered six-lag record targets: insulin x2 -12.3739/half +6.1870; NE x2 +4.3837/half -2.1918 (effects beyond lag six omitted).
- Current half-strength edit remains weak; extracting forward 18 NLL and exact candidate objects without verbose architecture/notes output.
- No weights changed. Python runs: 253; forward passes: 19.

## Run 254: half-strength correction fails factual criterion
- Forward 18 factual NLL original 1.63155439, edited 1.64171492, delta +0.01016053.
- Event deltas: finger +0.00322, lab +0.00084, arterial +0.01930, cuff +0.01331; MAP dominates degradation.
- Saved current candidate in recurrent_half_backup.npz; recovering separate drug components and replay interfaces before redesign. No weights changed.
- Python runs: 254; forward passes: 19.

## Run 255: separate recurrent readout routes
- Algebraically recovering glucose and MAP components of the rank-two correction.
- Inspecting multiplier-general rate feature contrasts and one-step evaluator before testing selective routing. No weights changed.
- Python runs: 255; forward passes: 19.

## Run 256: full-strength glucose-only recurrent candidate
- Split correction into glucose route (norm 118.13) and MAP route (norm 1169.18).
- Current feature contrasts are nonlinear: insulin x2/x3 ~0.304/0.323; NE x2/half ~0.03174/-0.00397, explaining weak half-dose response.
- Submitted core.b_in = original + glucose route, with MAP route disabled.
- Local multiplier checks are diagnostic only; official factual and rollout validation next.
- Python runs: 256; forward passes: 19.

## Run 257: slot-mean factual constraints
- Glucose-only candidate passes forward 20 factual, delta 0.00300080; forward 21 shifts insulin -10.203/+4.439, NE +0.327/-0.048.
- Fitting multiplier-normalized physical-rate contrasts with separate penalties on slot-mean and within-slot factual energy. Hypothesis: avoid penalizing cancellation directions that have little readout effect.
- No weights changed. Python runs: 257; forward passes: 21.

## Run 258: recover exact interfaces before further edits
- Slot-mean feature fits alone do not resolve dose-response asymmetry.
- Preserved the factual-validated glucose-only edit in recurrent_glucose_valid.npz.
- Recovering architecture, identification evidence and retained helper interfaces.
- Submitted weights unchanged. Python runs: 258; forward passes: 21.

## Run 259: compact interface inventory
- Run 258 output was truncated before the helper inventory.
- Record-based immediate and dynamic effects recovered; no medical assumptions used.
- Recovering compact helper and candidate metadata. No weights changed.
- Python runs: 259; forward passes: 21.

## Run 260: inspect recurrent calibration and centering
- Separate glucose and MAP routes are retained; investigating MAP asymmetry.
- Inspecting exact prediction helpers, route norms and factual feature means.
- No weights changed. Python runs: 260; forward passes: 21.

## Run 261: constrained MAP dose-feature fits
- Current features are centered; constant-bias correction is not justified.
- Fitting minimum-factual-energy MAP features with explicit multiplier constraints.
- Targets use record-estimated MAP impulse effect, not medical knowledge.
- No weights changed. Python runs: 261; forward passes: 21.

## Run 262: conservative MAP route
- Exact all-multiplier constraints produced unacceptable factual feature energy.
- Fitted selected dose constraints and capped factual feature SD at 0.50 MAP units locally.
- Checking insulin cross-talk and NE x0/.5/1.5/2/3 responses before submission.
- No weights changed. Python runs: 262; forward passes: 21.

## Run 263: submit conservative MAP trial
- Added capped, multiplier-constrained MAP route to the validated glucose route.
- Local NE x2/half arterial shifts: +0.3409/-0.1122; x3 still flattens.
- Local factual MAP RMS shift ~0.84; official eligibility remains unknown.
- Glucose-only fallback remains archived and recoverable.
- Python runs: 263; forward passes: 21.

## Run 264: recover after rejected MAP trial
- Forward 22 factual delta +0.00909 exceeds the +0.005 limit.
- Recovering the archived eligible glucose-only candidate and exact replay interfaces.
- Current trial is not acceptable for submission; no weights changed.
- Python runs: 264; forward passes: 22.

## Run 265: verify separable corrections
- Checking the glucose fallback and rejected MAP addition against their retained matrices.
- Recovering exact local cases and validation stays before attenuation tests.
- No weights changed. Python runs: 265; forward passes: 22.

## Run 266: reject inefficient MAP route; strengthen glucose modestly
- Forward 23 rejected MAP trial: insulin -10.779/+4.926; NE +1.113/-0.359.
- MAP improvement is too small for its +0.00909 total factual-loss cost.
- Submitted glucose-only correction at strength 1.15; MAP disabled.
- Recovering full-distribution readout interfaces for a better MAP route.
- Python runs: 266; forward passes: 23.

## Run 267: full-distribution MAP routing
- Forward 24 glucose strength 1.15 factual delta +0.00383 passes on the original 100-stay batch.
- Computing finite-difference value-probability Jacobians and minimum-Fisher-cost MAP routes.
- No autograd, backpropagation or weight optimiser used; submitted weights unchanged.
- Python runs: 267; forward passes: 24.

## Run 268: verified probability capture and MAP route fit
- Corrected latent-dimension handling; verified probability-derived means before fitting.
- Fitted minimum-categorical-distortion routes by finite differences and linear algebra only.
- Submitted glucose-only weights unchanged.
- Python runs: 268; forward passes: 24.

## Run 269: restart recovery
- Reading architecture, identification evidence and retained interfaces.
- Minimum-Fisher MAP routes remain locally tested only.
- Submitted glucose-only strength 1.15 edit unchanged.
- Python runs: 269; forward passes: 24.

## Run 270: bounded interface recovery
- Run 269 output truncated before fit inventory; retrieving compact objects and archives.
- No weights changed. Python runs: 270; forward passes: 24.

## Run 271: focused construction audit
- Retrieving injection formula and initial record-identification evidence.
- No weights changed. Python runs: 271; forward passes: 24.

## Run 272: compare minimum-distortion MAP routes
- Recovered retained Fisher-route fits and checked the recurrent injection factorization.
- Compared local factual mean disturbance and NE doubled/halved responses with the same dose feature.
- No weights changed; local results are not official factual or rollout validation.
- Python runs: 272; forward passes: 24.

## Runs 273-274: separate archived components and compare KL
- Run 273 stopped on a rank-one assertion before edits or probability tests.
- Run 274 subtracted the archived glucose component before checking the MAP factorization.
- Compared probability KL and NE dose responses on 320 local cases, including 296 outside the direction-fitting subset.
- No weights changed. Python runs: 274; forward passes: 24.

## Run 275: feature-aware factual distortion
- Direction-only Fisher routes had similar held-out KL; half strength cost approximately 0.0009 categorical nats locally.
- Computed scalar-slot finite differences and the induced probability Fisher matrix for a linear dose feature.
- Recovered record-based target metadata and a second factual-validation batch.
- No backpropagation, weight optimizer, or weight changes.
- Python runs: 275; forward passes: 24.

## Run 276: slot-aware dose-feature regression
- Glucose-only strength 1.15 passed a second 100-stay factual batch; computed pooled exact delta from forwards 24 and 25.
- Built dose-contrast features by scalar-slot finite differences, for x0/.5/1.5/2/3 of both drugs.
- Fit small linear dose-feature models using record-estimated MAP effects, factual probability distortion, and optional slot-energy penalties.
- Alternating local cases held out from fitting; candidate quadratic KL capped at 0.0012 on both halves.
- No weight changes, backpropagation, or weight optimizer.
- Python runs: 276; forward passes: 25.

## Run 277: exact local tests and bounded MAP trial
- Low-penalty feature fits overfit the local fitting half; rejected those routes.
- Exactly evaluated regularized feature candidates and full-local-sample refits across all five NE multipliers and insulin cross-talk.
- Any submitted trial has a recoverable glucose-only backup in map_feature_trial_backup.npz.
- Selected trial: ('full_fit', 0.01, 5.0). Official eligibility and rollout response remain unvalidated.
- Python runs: 277; forward passes: 25.

## Run 278: recover official-validation inputs
- Current MAP trial remains unvalidated; glucose-only backup is preserved.
- Recovered architecture, record-derived targets, archived components and validation stay IDs.
- No weights changed. Python runs: 278; forward passes: 25.

## Run 279: bounded validation recovery
- Run 278 output truncated before validation metadata; narrowed output.
- Record evidence supports persistent insulin/glucose and NE/MAP effects, subject to identification assumptions.
- No weights changed. Python runs: 279; forward passes: 25.

## Run 280: reject unstable MAP feature route
- Forward 26: edited factual NLL 5.24586 versus original 1.63155; catastrophic recurrent amplification.
- Restored exact glucose-only strength-1.15 backup from map_feature_trial_backup.npz.
- Local probability KL is insufficient for recurrent-edit eligibility; inspect lower-norm routes before retrying.
- Python runs: 280; forward passes: 26.

## Run 281: lower-norm regularized MAP trial
- Added half of retained ridge-1.0 minimum-Fisher MAP route to glucose strength 1.15.
- Addition norm approximately 294.64 versus 3414.85 for the catastrophic trial.
- Full-sequence factual and rollout validation required; no stability claim from norm alone.
- Backup and trial saved in map_regularized_trial.npz.
- Python runs: 281; forward passes: 26.

## Run 282: recover final calibration evidence
- Forward 28 current trial: insulin x2/half -11.952/+5.366; NE x2/half +0.665/-0.164.
- These are predictions, not drug-effect truth. Recovering record targets and forward 27 factual fit before retaining the MAP addition.
- No weights changed. Python runs: 282; forward passes: 28.

## Run 283: narrow eligibility and interface recovery
- Read architecture and early record identification evidence in run 282; output truncated before current factual fit.
- Retrieving forward 27 NLL, candidate keys and compact interfaces without repeating architecture.
- No weights changed. Python runs: 283; forward passes: 28.

## Run 284: prepare independent validation
- Forward 27 exact factual delta: +0.00486788045 nats on 4,751 readings; narrow pass on one 100-stay batch.
- Recovering disjoint tuning batches, cumulative record targets and exact archived edit components.
- No weights changed. Python runs: 284; forward passes: 28.

## Run 285: pooled validation and balanced strength trial
- Previous candidate passed all 400 tuning stays: paired value-NLL delta 0.003163118 over 18654 readings; stay-bootstrap 95% interval [0.0017847891050899794, 0.0045359666147188745].
- First-batch record convolution targets: insulin x2 approximately -10.37 to -11.39; NE x2 approximately +4.21 to +4.25.
- Archived eligible 400-stay fallback in validated_current400.npz.
- Submitted glucose strength 1.08 instead of 1.15 and 1.6 times the current MAP addition (full-route strength approximately 0.8).
- Trial needs official factual and dose-rollout validation. No gradients or weight optimizer.
- Python runs: 285; forward passes: 31.

## Run 286: balanced-trial eligibility recovery
- Forward 32 first-batch value-NLL delta approximately +0.00613; not eligible on that batch alone.
- Recovering exact archives, architecture and validation inputs before pooled validation or rollback; no weights changed.
- Python runs: 286; forward passes: 32.

## Run 287: balanced candidate, complete factual validation
- Forwards 32/34/35/36: paired 400-stay value-NLL delta 0.004253115, 18654 readings; bootstrap quantiles [2.5%,50%,95%,97.5%]=[0.002651140325587262, 0.0042566779481620865, 0.005619416365475324, 0.00586519285366469].
- Forward 33 predictions: insulin x2/half -11.712/+5.178; NE x2/half +0.926/-0.311. MAP remains substantially under-responsive relative to retained record estimates.
- Archived exact current weights and pooled validation; no weights changed.
- Python runs: 287; forward passes: 36.

## Run 288: safety-margin candidate
- Balanced candidate passes the observed 400-stay mean but its 97.5% stay-bootstrap bound exceeds +0.005; not selected as final.
- Submitted glucose strength 1.00 and MAP full-route strength 0.70 (1.40 times the validated fallback MAP component).
- Goal: insulin x2 nearer record-derived target, stronger MAP response than fallback, and a safer factual-loss margin.
- Archived in margin_candidate.npz; needs official factual and dose-rollout validation. No gradients or weight optimizer.
- Python runs: 288; forward passes: 36.

## Run 289: final-stage recovery
- Forward 38 safety-margin candidate: value-NLL delta approximately +0.00282 on 4664 readings.
- Recovering architecture, exact forwards 37/38, validation batches and archived components.
- No weights changed; margin candidate still requires pooled factual and rollout validation.
- Python runs: 289; forward passes: 38.

## Run 290: exact partial validation and record-evidence recovery
- Margin candidate forwards 37/38 pooled delta: 0.003947722 over 9415 readings from 200 stays.
- Retrieved remaining disjoint factual batches and prior record-identification evidence.
- No weights changed. Python runs: 290; forward passes: 38.

## Run 291: margin candidate complete factual validation and first rollout
- Factual files fwd_37/38/40/41: exact pooled 400-stay delta 0.003384797 over 18654 readings.
- Stay-bootstrap quantiles [2.5%,50%,95%,97.5%]: [0.0019512287514498496, 0.00338410632250692, 0.004614249016037766, 0.00487369511280955].
- Forward count 41, file fwd_42: insulin x2/half -10.879/+4.850; NE x2/half +0.854/-0.257.
- Insulin is near retained record-convolution targets; NE remains materially under-responsive.
- Archived exact validated weights; no weights changed, gradients, or weight optimizer.
- Python runs: 291; forward passes: 41 (one failed request created a folder-number offset).

## Run 292: restart recovery
- Forward 42 (folder fwd_43) confirms insulin response on a second batch (-9.287/+6.039), while NE remains weak (+0.508/-0.118).
- Recovering architecture, training-record evidence, and exact candidate components. No weights changed.
- Python runs: 292; forward passes: 42.

## Run 293: compact evidence recovery
- Previous recovery output was truncated; retrieved selected identification notes, candidate metadata, and rollout schemas without changing weights.
- Python runs: 293; forward passes: 42.

## Runs 294-295: complete unchanged-candidate rollout validation
- Run 294 confirmed exact weight equality to margin_validated400.npz, then stopped on a pandas method/column-name collision; no weights changed.
- Forwards 43/44 (folders fwd_44/fwd_45) completed all 400 tuning rollouts.
- Run 295 pooled reached stay-hours and compared against six-lag training-record impulse convolutions. These targets require causal identification assumptions and omit the longer tail; they are not observed truth.
- Saved final_rollout400_summary.csv and final_rollout400_record_comparison.csv.
- Python runs: 295; forward passes: 44.
        edit  mean_shift  stay_bootstrap_low  stay_bootstrap_high  empirical_six_lag_target  n_stay_hours  n_stays
insulin_half    5.460528            4.723981             6.161704                  6.360204          2009      214
  insulin_x2   -9.560353          -10.235576            -8.911773                -12.720408          2009      214
     ne_half   -0.165227           -0.232646            -0.102521                 -2.084802          1080       96
       ne_x2    0.679243            0.430757             0.964145                  4.169603          1080       96

## Run 296: audit discrepancy in empirical rollout targets
- Pooled current rollouts: insulin x2/half -9.560353/+5.460528; NE x2/half +0.679243/-0.165227.
- Run-6 six-lag convolution on actual reached readout rows yields insulin -12.720408/+6.360204 and NE +4.169603/-2.084802, but does not reproduce the older run-132 first-batch estimates.
- Recovering exact run-132 coefficients and averaging convention before drawing a calibration conclusion; no weights changed.
- Python runs: 296; forward passes: 44.

## Run 297: final calibration and submission audit
- Recovered architecture, retained convolution inputs, pooled rollout summaries, exact submitted-weight archive comparison, and available forward metadata.
- No weights changed. Current candidate retains its previously measured 400-stay factual delta of +0.003384797 nats per reading.
- Record-derived targets remain estimates requiring identification assumptions, not observed intervention truth.
- Python runs: 297; forward passes: 44.

## Run 298: same-row target and archived-candidate audit
- Recomputed six-lag physical-dose targets on identical reached rows; compared first-80, first-100 and full-400 cohorts under hour and stay weighting.
- Inspected exact archived stronger-insulin and margin candidates; no weights changed.
- Current factual delta remains +0.003384797 nats per reading. The previously validated glucose-1.15/MAP-0.5 candidate had +0.003163118, with stay-bootstrap 97.5% bound +0.004535967.
- Python runs: 298; forward passes: 44.

## Run 299: resolve averaging discrepancy and compare eligible candidates
- Run-132 targets used equal-stay averaging; later summaries pooled stay-hours. Same-row convolution discrepancies are at numerical precision only.
- Full-400 six-lag targets: hour-weighted insulin x2/half -12.720408/+6.360204 and NE x2/half +4.169603/-2.084802; stay-weighted -10.933000/+5.466500 and +4.090764/-2.045382.
- Independently reconstructed stronger-insulin candidate factual validation from fwd_27/29/30/31: delta 0.003163117830 over 18654 readings; bootstrap quantiles [0.0018336174687510057, 0.00316119528937411, 0.004307877545840386, 0.0045213344637845465].
- Exact stronger-insulin archive restored: True. Only core.b_in.weight is edited; no gradients or optimizer used.
- Final selection will compare identical 400-stay rollout cohorts, prioritizing the stated main insulin readout while documenting the MAP tradeoff.
- Both candidate weights preserved in final_switch_audit.npz.
- Python runs: 299; forward passes: 44.


## Run 300: final unchanged-candidate audit and partial rollout pooling
- Edited tensors: ['core.b_in.weight']; core.b_in.weight SHA256: 70f5f731e534fc9e4c69574865c0c2d87c637ccd8d7d7943f86b2e0f5b1f49ce.
- Independent 400-stay factual validation: paired value-NLL delta 0.003163117830 over 18654 readings; bootstrap quantiles [2.5%,50%,95%,97.5%]=[0.0017824940644117608, 0.003183081857872992, 0.004338861439409167, 0.00457841350274192].
- Pooled the stronger-insulin candidate's first 300 tuning stays and the margin candidate's complete 400-stay rollouts; saved final_comparison_partial300.csv.
                   cohort    model         edit  mean_shift   shift_sum  n_stay_hours  n_stays
stronger_insulin_first300 original insulin_half    1.175905   1732.1075          1473      156
stronger_insulin_first300 original   insulin_x2   -1.820754  -2681.9704          1473      156
stronger_insulin_first300 original      ne_half   -0.040755    -34.1526           838       75
stronger_insulin_first300 original        ne_x2    0.270446    226.6338           838       75
stronger_insulin_first300   edited insulin_half    6.289322   9264.1713          1473      156
stronger_insulin_first300   edited   insulin_x2  -10.712349 -15779.2897          1473      156
stronger_insulin_first300   edited      ne_half   -0.125948   -105.5445           838       75
stronger_insulin_first300   edited        ne_x2    0.606619    508.3471           838       75
            margin_all400 original insulin_half    1.100821   2211.5500          2009      214
            margin_all400 original   insulin_x2   -1.785651  -3587.3736          2009      214
            margin_all400 original      ne_half   -0.036495    -39.4143          1080       96
            margin_all400 original        ne_x2    0.250217    270.2348          1080       96
            margin_all400   edited insulin_half    5.460528  10970.2003          2009      214
            margin_all400   edited   insulin_x2   -9.560353 -19206.7493          2009      214
            margin_all400   edited      ne_half   -0.165227   -178.4456          1080       96
            margin_all400   edited        ne_x2    0.679243    733.5820          1080       96
- Recovered the remaining matched 100-stay cohort for one final dose-edit validation. No further weight changes are planned.
- Norepinephrine remains under-responsive; record-convolution targets are identification-dependent estimates, not observed counterfactual truth. Additional intervention factors, start times, cross-sensor readouts and all-output loss are not independently certified by this audit.
- No weights changed, gradients or weight optimizer used. Python runs: 300/300; forward passes: 47/50.
- Pending final request: {"type": "dose_edit", "model": "both", "split": "tuning", "stays": [2020000304, 2020000320, 2020000336, 2020000352, 2020000368, 2020000384, 2020000305, 2020000321, 2020000337, 2020000353, 2020000369, 2020000385, 2020000306, 2020000322, 2020000338, 2020000354, 2020000370, 2020000386, 2020000307, 2020000323, 2020000339, 2020000355, 2020000371, 2020000387, 2020000308, 2020000324, 2020000340, 2020000356, 2020000372, 2020000388, 2020000309, 2020000325, 2020000341, 2020000357, 2020000373, 2020000389, 2020000310, 2020000326, 2020000342, 2020000358, 2020000374, 2020000390, 2020000311, 2020000327, 2020000343, 2020000359, 2020000375, 2020000391, 2020000312, 2020000328, 2020000344, 2020000360, 2020000376, 2020000392, 2020000313, 2020000329, 2020000345, 2020000361, 2020000377, 2020000393, 2020000314, 2020000330, 2020000346, 2020000362, 2020000378, 2020000394, 2020000315, 2020000331, 2020000347, 2020000363, 2020000379, 2020000395, 2020000300, 2020000316, 2020000332, 2020000348, 2020000364, 2020000380, 2020000396, 2020000301, 2020000317, 2020000333, 2020000349, 2020000365, 2020000381, 2020000397, 2020000302, 2020000318, 2020000334, 2020000350, 2020000366, 2020000382, 2020000398, 2020000303, 2020000319, 2020000335, 2020000351, 2020000367, 2020000383, 2020000399]}
