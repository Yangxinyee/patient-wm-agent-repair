# World-model repair notebook

## Initial plan
- Read architecture and record definitions before choosing an identification strategy.
- Infer drug effects only from the simulated ICU records; impose no medical assumptions.
- Use no gradients or weight optimiser.
- Validate weight edits against original factual NLL and paired dose rollouts.
- Required factual NLL degradation: at most 0.005 nats per reading.
- Python runs: 1. Forward requests: 0.

## Run 2: architecture
Drug atoms have their own action encoder, but enter posterior and recurrent updates. Value heads also use last-reading registers and code-specific bin transitions. No causal conclusion yet. Full records loaded for timing and identification analysis.
Python runs: 2. Forward requests: 0.

## Run 3: hourly panel
Readings occur on integer hours; infusions start at hour + 0.0833 and last one hour. Built full panels with absent infusion=0, prior reading registers, ages, and next-hour outcomes. Initial regressions are diagnostics, not yet accepted causal estimates.
Validation IDs chosen randomly once (seed 731), 100 tuning stays.
Python runs: 3. Forward requests: 0.

## Run 4: selection and threshold identification
Unadjusted insulin/glucose slope +23.36, adjusted about -1.92 per unit/hour. NE/MAP slope changes from -30.02 raw to -1.90 with current reading and +2.25 with history. These are not accepted as causal. Insulin dose thresholds near 150/180/220 motivate fuzzy RD. Test independent lab outcomes and observation availability to diagnose selection artifacts.
Python runs: 4. Forward requests: 0.

## Forward 1: original rollout, first 100 tuning stays
Insulin x2 -1.847; insulin half +0.777; NE x2 -0.498; NE half -0.600. Insulin reaches 554 stay-hours/60 stays; NE 281/26. These are predictions, not truth.
RD roughly -2 per additional insulin unit/hour, with serious observation jump at 150; prefer thresholds 180/220 and check temporal persistence. Boundary observations are mixtures because measurements are rounded.
Python runs: 5. Forward requests: 1.

## Run 5–6: dynamic identification
Pooled insulin RD excludes rounded cutoff values at 180/220. Immediate glucose jump -2.2131 (SE .1583), dose jump 1.0361. Reduced-form outcome discontinuity fades over six hours; downstream dosing also changes. Joint impulse fit beta=-2.2778, persistence=.85743; provisional, not exact truth. NE polynomial-history coefficients remain about +3.09, but dose-policy nonlinearity may bias them. Cross-fitted flexible nuisance models now test residual-dose effects.
NE DML beta=3.248868, cluster SE=0.096302.
Python runs: 6. Forward requests: 1.

## Run 7: restart and factual baseline
Original factual value NLL on the fixed 100-stay tuning sample: 1.63461 (4751 readings); allowed sample ceiling 1.63961. Recovering architecture and retained analysis state before choosing an edit.
Python runs: 7. Forward requests: 2.

## Run 8: action pathway and identification audit
Architecture recovered: action-specific encoder feeds posterior, prelude, and recurrent state; value heads also have same-code last-reading registers. Inspecting exact dose support, RD results, and layer dimensions before surgery.
Python runs: 8. Forward requests: 2.

## Run 9: compact units and causal audit
Retrieved dynamic identification, dose units/support, exact action/posterior shapes, and head-input configuration. No weights edited yet.
Python runs: 9. Forward requests: 2.

## Run 10: states and target scale
Loaded original 4356 bundle states for tuning stays 2020000000..99 and all original tensors. Computed provisional fixed-dose schedule responses over several persistence values; these are statistical targets, not hidden truth. No weights edited.
Python runs: 10. Forward requests: 3.

## Run 11: local reading decoder reconstruction
Enumerated register layouts and activation types. Best: gelu, (6, 3, (0,)); decoder RMSE 1.5651. Provisional insulin x2 target on recorded reached hours: -13.586; half target +6.793, assuming beta=-2.2778 and persistence=.8574.
Python runs: 11. Forward requests: 3.

## Run 12: policy residual probes
Fitted training-record dose-policy nuisance models (260000 sampled intervals). Cross-stay ridge probes test residual dose and raw-dose recovery from reading-head features; separately quantified prior-sampling noise. Probe results: [(0.1, [{'drug': 'insulin', 'residual_SD': 0.4592, 'residual_R2': -0.0309, 'raw_dose_R2': 0.7405, 'other_event_leakage_RMS': 0.0716}, {'drug': 'norepinephrine', 'residual_SD': 0.073, 'residual_R2': -0.0588, 'raw_dose_R2': 0.5973, 'other_event_leakage_RMS': 0.0171}]), (1.0, [{'drug': 'insulin', 'residual_SD': 0.4592, 'residual_R2': -0.03, 'raw_dose_R2': 0.7384, 'other_event_leakage_RMS': 0.067}, {'drug': 'norepinephrine', 'residual_SD': 0.073, 'residual_R2': -0.0447, 'raw_dose_R2': 0.6053, 'other_event_leakage_RMS': 0.0166}]), (10.0, [{'drug': 'insulin', 'residual_SD': 0.4592, 'residual_R2': -0.028, 'raw_dose_R2': 0.7306, 'other_event_leakage_RMS': 0.0539}, {'drug': 'norepinephrine', 'residual_SD': 0.073, 'residual_R2': -0.0171, 'raw_dose_R2': 0.6227, 'other_event_leakage_RMS': 0.0132}]), (100.0, [{'drug': 'insulin', 'residual_SD': 0.4592, 'residual_R2': -0.0198, 'raw_dose_R2': 0.7251, 'other_event_leakage_RMS': 0.0339}, {'drug': 'norepinephrine', 'residual_SD': 0.073, 'residual_R2': 0.0015, 'raw_dose_R2': 0.6356, 'other_event_leakage_RMS': 0.0079}]), (1000.0, [{'drug': 'insulin', 'residual_SD': 0.4592, 'residual_R2': -0.0094, 'raw_dose_R2': 0.7131, 'other_event_leakage_RMS': 0.0186}, {'drug': 'norepinephrine', 'residual_SD': 0.073, 'residual_R2': 0.0017, 'raw_dose_R2': 0.6434, 'other_event_leakage_RMS': 0.0037}])]
Python runs: 12. Forward requests: 3.

## Run 13: test earlier representations
Initial event-gated head probes failed held-out residual-dose recovery. Testing ungated probes across query, hidden head, full state, and posterior to locate the bottleneck; inspected atom encoder tables.
Ungated results: [{'representation': 'query', 'alpha': 1.0, 'residual_R2': [-0.106, -0.006], 'raw_R2': [0.794, 0.679]}, {'representation': 'query', 'alpha': 30.0, 'residual_R2': [-0.015, -0.01], 'raw_R2': [0.806, 0.663]}, {'representation': 'query', 'alpha': 300.0, 'residual_R2': [0.005, -0.005], 'raw_R2': [0.797, 0.652]}, {'representation': 'head_input', 'alpha': 1.0, 'residual_R2': [-0.05, -0.025], 'raw_R2': [0.803, 0.672]}, {'representation': 'head_input', 'alpha': 30.0, 'residual_R2': [-0.011, -0.016], 'raw_R2': [0.803, 0.646]}, {'representation': 'head_input', 'alpha': 300.0, 'residual_R2': [0.001, -0.009], 'raw_R2': [0.785, 0.64]}, {'representation': 'head_hidden', 'alpha': 1.0, 'residual_R2': [-0.091, -0.168], 'raw_R2': [0.771, 0.572]}, {'representation': 'head_hidden', 'alpha': 30.0, 'residual_R2': [-0.035, -0.06], 'raw_R2': [0.78, 0.618]}, {'representation': 'head_hidden', 'alpha': 300.0, 'residual_R2': [-0.014, -0.014], 'raw_R2': [0.768, 0.636]}, {'representation': 'out', 'alpha': 1.0, 'residual_R2': [-0.021, 0.001], 'raw_R2': [0.815, 0.68]}, {'representation': 'out', 'alpha': 30.0, 'residual_R2': [-0.0, 0.0], 'raw_R2': [0.81, 0.658]}, {'representation': 'out', 'alpha': 300.0, 'residual_R2': [0.005, -0.004], 'raw_R2': [0.798, 0.648]}, {'representation': 'mean_h', 'alpha': 1.0, 'residual_R2': [-0.024, 0.001], 'raw_R2': [0.817, 0.692]}, {'representation': 'mean_h', 'alpha': 30.0, 'residual_R2': [0.003, 0.004], 'raw_R2': [0.811, 0.668]}, {'representation': 'mean_h', 'alpha': 300.0, 'residual_R2': [0.006, 0.0], 'raw_R2': [0.797, 0.655]}, {'representation': 'z_posterior', 'alpha': 1.0, 'residual_R2': [0.143, 0.023], 'raw_R2': [0.416, 0.277]}, {'representation': 'z_posterior', 'alpha': 30.0, 'residual_R2': [0.041, 0.014], 'raw_R2': [0.329, 0.164]}, {'representation': 'z_posterior', 'alpha': 300.0, 'residual_R2': [0.021, 0.015], 'raw_R2': [0.252, 0.139]}, {'representation': 'all_state', 'alpha': 1.0, 'residual_R2': [0.257, -0.033], 'raw_R2': [0.881, 0.692]}, {'representation': 'all_state', 'alpha': 30.0, 'residual_R2': [0.068, 0.02], 'raw_R2': [0.836, 0.691]}, {'representation': 'all_state', 'alpha': 300.0, 'residual_R2': [0.034, 0.023], 'raw_R2': [0.822, 0.673]}]
Python runs: 13. Forward requests: 3.

## Run 14: restart recovery and intervention-path hypothesis
Downstream probes encode policy-associated dose but almost no residual dose; a head-only edit is unlikely to recover intervention sensitivity reliably. Recovering exact recurrent/action architecture and retained statistical objects before designing a dose-specific, factually compensated edit. No weights edited.
Python runs: 14. Forward requests: 3.

## Run 15: posterior-edit feasibility
Posterior receives context and action jointly. Candidate repair: recover dose-policy innovations in its hidden features and route them through a low-rank posterior-mean edit, with non-action updates constrained unchanged. Need exact input reconstruction and held-out innovation recovery first.
Python runs: 15. Forward requests: 3.

## Run 16: exact posterior reconstruction prerequisites
Retrieve latent-layer names, helper signatures, policy covariates, categorical defaults, and start/END action-bundle composition. Require a match to saved posterior states before using local features for an edit. No weights edited.
Python runs: 16. Forward requests: 3.

## Run 17: single-start posterior matching
Reconstructed action-only posterior inputs; searched bin boundaries, unit mappings, activations and gap transforms. Best candidates: {'insulin': {'drug': 'insulin', 'rmse': 0.16355973482131958, 'unit_id': 0, 'rate_unit_id': 0, 'bin_side': 'left', 'activation': 'gelu', 'gap_style': 'log1p'}, 'norepinephrine': {'drug': 'norepinephrine', 'rmse': 0.08330594003200531, 'unit_id': 4, 'rate_unit_id': 3, 'bin_side': 'left', 'activation': 'gelu', 'gap_style': 'log'}}
No weights edited. Python runs: 17. Forward requests: 3.

## Run 18: reconstruction error audit
Single-start posterior errors remain too large for a verified reconstruction. Tested key-quality embeddings, empty observation encoding, dual encoding, and possible prior-mean addition. Best candidates: {'insulin': {'drug': 'insulin', 'rmse': 3.339840623084456e-05, 'unit_id': 3, 'rate_unit_id': 2, 'kq_id': 0, 'e_mode': 'empty', 'add_prior': False}, 'norepinephrine': {'drug': 'norepinephrine', 'rmse': 4.155545320827514e-05, 'unit_id': 4, 'rate_unit_id': 3, 'kq_id': 0, 'e_mode': 'empty', 'add_prior': False}}
No weights edited. Python runs: 18. Forward requests: 3.

## Run 19: END and whole-sequence reconstruction
Single starts match after including kq[0] and nonzero empty observation encoding. Gap=log; END={'end_numeric': 'absent', 'end_units': 'code', 'rmse': 0.04775621369481087}; atom-count mismatches=0. Whole-sequence posterior RMSE=0.1773110.
No weights edited. Python runs: 19. Forward requests: 3.

## Run 20: restart recovery
Full posterior reconstruction still disagrees on reading bundles and some END bundles; recovering architecture, causal evidence, and retained objects before any edit. No tensors edited. Python runs: 20. Forward passes: 3.

## Run 21: compact kernel inventory
Recovering helper signatures, array shapes, vocab structure, and encoder tensors after the previous output was truncated. No edits. Python runs: 21; forward passes: 3.

## Run 22: isolate reconstruction errors
Inspect exact embedding tables, existing encoding constants, and bundle-specific errors. No edits. Python runs: 22; forward passes: 3.

## Run 23: measurement embedding search
{'glucose_finger': [((1, 1, 1), 0.14344097673892975), ((3, 1, 1), 0.14357315003871918), ((4, 1, 1), 0.14490433037281036)], 'glucose_lab': [((1, 1, 0), 0.14021655917167664), ((4, 1, 0), 0.14032329618930817), ((4, 1, 3), 0.14146500825881958)], 'map_arterial': [((2, 3, 1), 0.1017359048128128), ((2, 3, 2), 0.10185274481773376), ((2, 3, 3), 0.10189926624298096)], 'map_cuff': [((2, 2, 0), 0.11021342873573303), ((2, 0, 0), 0.11098862439393997), ((2, 1, 0), 0.11161728203296661)]}
No weight edits. Python runs: 23; forward passes: 3.

## Run 24: absent-attribute audit
{'glucose_finger': [((3, 0, 0, -1), 0.1589316874742508), ((3, 0, 0, 0), 0.1589316874742508), ((3, 0, -1, 0), 0.1589316874742508)], 'glucose_lab': [((1, 0, -1, -1), 0.1447429656982422), ((1, 0, -1, 0), 0.1447429656982422), ((1, 1, -1, 0), 0.1447429656982422)], 'map_arterial': [((2, 1, -1, -1), 0.10221867263317108), ((2, 1, -1, 0), 0.10221867263317108), ((2, 1, 0, -1), 0.10221867263317108)], 'map_cuff': [((2, 1, -1, -1), 0.11098864674568176), ((2, 1, -1, 0), 0.11098864674568176), ((2, 1, 0, -1), 0.11098864674568176)]}
No edits. Python runs: 24; forward passes: 3.

## Run 26: assembly implementation audit
Activation and unit searches did not solve observation mismatch. Comparing fresh versus retained atoms/contexts and singleton predictions. No edits. Python runs: 26; forward passes: 3.

## Run 27: empty-action convention test
{'glucose_finger': {'zero': 0.17373116314411163, 'encoded_empty': 0.1684964895248413, 'observation_encoder_empty': 0.17108602821826935}, 'glucose_lab': {'zero': 0.20552821457386017, 'encoded_empty': 0.20301900804042816, 'observation_encoder_empty': 0.19132423400878906}, 'map_arterial': {'zero': 0.16247990727424622, 'encoded_empty': 0.093088798224926, 'observation_encoder_empty': 0.142836332321167}, 'map_cuff': {'zero': 0.1760014146566391, 'encoded_empty': 0.1547107845544815, 'observation_encoder_empty': 0.1544232964515686}}
No edits. Python runs: 27; forward passes: 3.

## Run 28: recover architecture and causal evidence
Empty-action alternatives did not verify the observation posterior. Recovering architecture, early causal estimates, and persistent objects before choosing a drug-bundle-specific edit. No weights edited.
Python runs: 28; forward passes: 3.

## Run 29: compact recovery
Architecture confirms the drug-specific action-to-posterior pathway. Earlier insulin RD and NE residual-dose estimates differ substantially from original rollouts, but remain provisional. Recovering retained statistical and encoding objects; no edits.
Python runs: 29; forward passes: 3.

## Run 30: targeted posterior recovery
Reading-head dose-innovation probes failed. Inspecting action-specific posterior features and exact saved-state format to test a factually centred latent correction. No edits.
Python runs: 30; forward passes: 3.

## Run 31: verified posterior feature test
Audited posterior reconstruction by bundle type and tested cross-stay innovation recovery on rows with posterior RMSE below .002.
Verified train/test: 100/22; probe results: [].
No weights edited. Python runs: 31; forward passes: 3.

## Run 32: ending audit and approximate innovation probes
Standalone drug starts are numerically verified, but repeated-hour bundles with END atoms are not. Tested larger approximate hidden-feature probes with explicit non-action zero constraints; diagnostic only.
Results: [{'other_weight': 1.0, 'alpha': 1.0, 'action_R2': [0.483, 0.01], 'non_action_RMS': [0.09759999811649323, 0.010499999858438969], 'verified_action_RMSE': [0.5036, 0.0634]}, {'other_weight': 1.0, 'alpha': 10.0, 'action_R2': [0.4446, 0.0154], 'non_action_RMS': [0.08560000360012054, 0.009100000374019146], 'verified_action_RMSE': [0.4947, 0.063]}, {'other_weight': 1.0, 'alpha': 100.0, 'action_R2': [0.3516, 0.0148], 'non_action_RMS': [0.06300000101327896, 0.007000000216066837], 'verified_action_RMSE': [0.462, 0.0628]}, {'other_weight': 10.0, 'alpha': 1.0, 'action_R2': [0.2872, 0.0058], 'non_action_RMS': [0.028699999675154686, 0.003700000001117587], 'verified_action_RMSE': [0.4776, 0.0641]}, {'other_weight': 10.0, 'alpha': 10.0, 'action_R2': [0.2742, 0.0114], 'non_action_RMS': [0.026799999177455902, 0.003599999938160181], 'verified_action_RMSE': [0.4759, 0.0638]}, {'other_weight': 10.0, 'alpha': 100.0, 'action_R2': [0.2279, 0.0173], 'non_action_RMS': [0.021299999207258224, 0.0031999999191612005], 'verified_action_RMSE': [0.4671, 0.0631]}].
No edits. Python runs: 32; forward passes: 3.

## Run 33: targeted encoding convention tests
END routing results: [('absent', 'action', 0.02602, 0.046565, 122), ('absent', 'observation', 0.000795, 0.001491, 1263), ('absent', 'ignored', 0.019119, 0.027057, 122), ('retained', 'action', 0.02624, 0.172787, 122), ('retained', 'observation', 0.007759, 0.01513, 122), ('retained', 'ignored', 0.019119, 0.027057, 122)].
Measurement kind/quality results: [('glucose_finger', [(0, 1, 0.16519242525100708), (0, 3, 0.16523434221744537), (0, 0, 0.16614624857902527)]), ('map_arterial', [(2, 0, 0.14014655351638794), (0, 0, 0.14225302636623383), (2, 1, 0.14460410177707672)]), ('map_cuff', [(0, 0, 0.2047487497329712), (2, 0, 0.2134970724582672), (0, 1, 0.21898208558559418)])].
Unchanged posterior basis recovered some insulin innovation but no reliable NE innovation. No edits. Python runs: 33; forward passes: 3.

## Run 34: corrected END routing and action-dose bottleneck
END tests: [('code', 2, 0.0014906, 122), ('code', 0, 0.0025535, 122), ('none', 2, 6.61e-05, 1489), ('none', 0, 0.0016234, 122)]; best convention ('none', 2).
Measurement value tests: [('glucose_finger', [('recorded', 'zero', 0.166146), ('recorded', 'encoded', 0.168488), ('bin_centre', 'zero', 0.171227), ('bin_centre', 'encoded', 0.173337)]), ('map_arterial', [('recorded', 'encoded', 0.083889), ('bin_centre', 'encoded', 0.086699), ('bin_centre', 'zero', 0.140569), ('recorded', 'zero', 0.142253)]), ('map_cuff', [('bin_centre', 'encoded', 0.192651), ('recorded', 'encoded', 0.195937), ('recorded', 'zero', 0.204749), ('bin_centre', 'zero', 0.207157)])].
Tested raw-dose and innovation recovery from incoming action vectors. No edits. Python runs: 34; forward passes: 3.

## Run 35: recover causal evidence before editing
END convention now numerically verifies drug bundles. Action encoding predicts raw dose better than condition-adjusted dose; this alone is not a causal correction. Recovering architecture and early record-based estimates. No edits. Python runs: 35; forward passes: 3.

## Run 36: compact state recovery
Early records imply insulin beta about -2.28 with persistence .857; original insulin x2 rollout -1.85 versus provisional record-based -13.59. NE residual-dose estimate +3.25 remains less securely identified. Recovering retained fitting objects without repeating architecture. No edits. Python runs: 36; forward passes: 3.

## Run 37: targeted action-path inventory
Full panels, causal models, original factual results and GPU states persist. Narrowing recovery to action/latent features and exact tensor names so an actual reversible edit can be tested. No edits. Python runs: 37; forward passes: 3.

## Run 38: factual-centred contrast design
Plan: fit a linear posterior-hidden feature with near-zero recorded-dose output and dose-proportional contrasts on synthetically re-encoded x2/half drug bundles. This is a statistical feature fit, not gradient training. Need exact helper conventions before constructing it. No edits. Python runs: 38; forward passes: 3.

## Run 39: synthetic dose-contrast features
Numerically verified drug bundles and fitted posterior-hidden features using recorded-dose zero constraints and synthetic x2/half dose contrasts. Diagnostics printed; no weights edited. Python runs: 39; forward passes: 3.

## Run 40: recurrent forward-only reconstruction
Drug posterior exact on all 1497 rows (RMS 6.6e-5). Existing hidden contrast features are weak: insulin slope .31 at factual RMS .47; stronger zero constraints reduce slope. Reconstructed recurrent block and compared memory/age conventions against saved GPU states; diagnostics printed. No edits. Python runs: 40; forward passes: 3.

## Run 41: restart recovery
Drug posterior is verified, but recurrent reconstruction remains inaccurate. Recovering architecture, causal evidence, and retained objects before an actual reversible edit. No weights edited. Python runs: 41; forward passes: 3.

## Run 42: compact restart inventory
Architecture recovered; retrieving retained statistical fits, encoding helpers, and states without repeating the full model listing. No edits. Python runs: 42; forward passes: 3.

## Run 43: targeted helper and effect diagnostics
Retained panels, GPU states, and synthetic action contrasts are available. Inspecting exact helpers and effect identification before a reversible posterior-path edit. No edits. Python runs: 43; forward passes: 3.

## Run 45: forward-only latent sensitivity
Computed central-difference posterior-latent sensitivity to next glucose and MAP readings, using the approximate local core only as an edit-direction diagnostic. No gradients or weight edits. Python runs: 45; forward passes: 3.

## Run 46: alternative prelude/update contrast probe
Posterior-latent sensitivity was too weak for a credible edit. Tested factual-zero dose-contrast regressions on prelude features and direct update sensitivities. No weights edited. Python runs: 46; forward passes: 3.

## Run 47: first reversible insulin trial
All-slot factual-zero / synthetic-dose-contrast linear regression fitted to prelude states. Selected anchor=10.0, insulin x2 slope=0.169392. Added rank-one core.b_in.weight correction, strength=0.75, Frobenius norm=297.520844. Direction lowers next glucose with approximately zero MAP response locally. Not accepted until GPU factual and rollout validation. Python runs: 47; forward passes: 3.

## Run 48: restart and trial-validation setup
The core.b_in.weight trial is not accepted: local direction is encouraging but factual leakage and asymmetric dose contrasts require GPU validation. Recovered architecture, early evidence and tuning identifiers. Python runs: 48; forward passes: 3.

## Forward 4–5 and Run 49: first trial validation
On 50 fixed tuning stays, factual NLL original=1.64063, trial=1.64660 (+0.00597: fails limit). Paired rollouts original/trial: insulin x2 -1.672/-6.811; insulin half +0.784/+2.535; NE x2 -0.548/-0.453; NE half -0.726/-0.831. Insulin direction works but requires better selectivity or factual repair; trial not accepted. Saved trial weights in kernel for reversible comparisons. Python runs: 49; forwards: 5.

## Run 50: targeted coefficient recovery
Prior recovery output was truncated. Restricted inspection to retained arrays and trial helpers before testing factual-innovation centering. No new edits. Python runs: 50; forwards: 5.

## Run 51: dose-innovation feature comparison
Closed-form statistical fits replace factual-zero targets with record-based action-dose innovations. No additional weight edits. Python runs: 51; forwards: 5.

## Run 52: feature-unit audit
Run 51 contrast diagnostics omitted standardization of raw prelude deltas. Corrected diagnostics and numerically recovered the trial rank-one formula. No new weight edits. Python runs: 52; forwards: 5.

## Run 53: restart recovery and acceptance criterion
Run 52 corrected feature scaling but did not establish a better edit. The existing core.b_in.weight trial remains unaccepted because its 50-stay factual NLL increased by 0.00597. Recovering architecture and retained objects before improving selectivity and testing both dose directions. No weights changed. Python runs: 53; forwards: 5.

## Run 54: causal evidence and factual-calibration inventory
Architecture confirms the current b_in edit modifies recurrent updates. Recovering record-based causal estimates, value-head tensor structure, and retained helpers to distinguish systematic factual bias from state distortion. No weights changed. Python runs: 54; forwards: 5.

## Run 55: factual-error localization
Compared paired original/trial means and NLL by event and time, and recovered the value-head helper dependencies. Record evidence supports insulin beta about -2.28 and provisional persistence .857; NE beta +3.25 needs stronger identification checks. No weights changed. Python runs: 55; forwards: 5.

## Run 56: admission-leakage nullspace test
Run 55 localized most trial NLL degradation to initial readings (glucose mean shifts +11/+20), not later factual forecasts. Constructed initial prelude features from original GPU posterior states and admission atoms; tested a rank-one input projection orthogonal to those features. No weights changed. Python runs: 56; forwards: 5.

## Run 57: constrained, unit-audited feature refits
Initial prelude has rank 4. Euclidean projection retained drug contrasts but was not accepted without a consistent factual-unit audit. Fitted closed-form two-output regressions with exact admission-null constraints, 80 fitting stays and 20 held-out stays; compared factual-zero and record-based innovation targets. Metrics: {('zero', 1.0): {'heldout_slot_rms': 0.2738265885911834, 'heldout_mean_rms': 0.2710827714511677, 'initial_rms': 8.292466768586752e-14, 'norm': 78.75577831207558, "('insulin', 2.0)": 0.4277882349997429, "('insulin', 0.5)": 0.36430773148947077, "('norepinephrine', 2.0)": 0.44816217033937794, "('norepinephrine', 0.5)": 0.13105941732850884}, ('zero', 10.0): {'heldout_slot_rms': 0.10106763445536245, 'heldout_mean_rms': 0.09987375315108506, 'initial_rms': 6.182895718594179e-14, 'norm': 39.820792483251445, "('insulin', 2.0)": 0.28130826817906374, "('insulin', 0.5)": 0.15675290351140922, "('norepinephrine', 2.0)": 0.44207759581503586, "('norepinephrine', 0.5)": 0.017280532549281236}, ('zero', 50.0): {'heldout_slot_rms': 0.037012737787550605, 'heldout_mean_rms': 0.03651322170367755, 'initial_rms': 2.4370849511692395e-14, 'norm': 16.097058498727602, "('insulin', 2.0)": 0.17538774551842928, "('insulin', 0.5)": 0.04189366264765239, "('norepinephrine', 2.0)": 0.3550545947978608, "('norepinephrine', 0.5)": 0.05671624298711782}, ('innovation', 1.0): {'heldout_slot_rms': 0.29670664731603336, 'heldout_mean_rms': 0.29402975820129124, 'initial_rms': 1.2206880698491195e-13, 'norm': 80.26858475589265, "('insulin', 2.0)": 0.4354821909919466, "('insulin', 0.5)": 0.41470753925118037, "('norepinephrine', 2.0)": 0.44214411864657455, "('norepinephrine', 0.5)": 0.1665920096879329}, ('innovation', 10.0): {'heldout_slot_rms': 0.18906080400350042, 'heldout_mean_rms': 0.188118794102396, 'initial_rms': 6.551482416451658e-14, 'norm': 52.16381770740218, "('insulin', 2.0)": 0.315645628664578, "('insulin', 0.5)": 0.33220903476339086, "('norepinephrine', 2.0)": 0.4387771209755884, "('norepinephrine', 0.5)": 0.1381956534689569}, ('innovation', 50.0): {'heldout_slot_rms': 0.16985576281511747, 'heldout_mean_rms': 0.1691551121442019, 'initial_rms': 3.7608876652477593e-14, 'norm': 43.93727089382375, "('insulin', 2.0)": 0.23142634479597984, "('insulin', 0.5)": 0.3034746278883954, "('norepinephrine', 2.0)": 0.3707051724009136, "('norepinephrine', 0.5)": 0.1849513119527869}}
No weights changed. Python runs: 57; forwards: 5.

## Run 58: installed admission-null insulin candidate
Projection alone failed the unit-correct factual audit. Selected constrained factual-zero fit anchor=10, with exact initial-prelude null constraints. Normalized its insulin-x2 contrast to the old trial (factor 3.6511856), then doubled trial strength. Edit formula: original b_in + .75*2*outer(trial_direction47, normalization58*constrained_fits57[('zero',10)][:,0]). Delta norm=312.882. Saved /work/insulin_candidate58.npz. Candidate is NOT accepted until GPU factual and paired rollout validation. Provisional training-record insulin targets: x2 about -13.6, half about +6.8. NE identification/persistence remains to audit. Python runs: 58; forwards: 5.

## Run 59: factual pass and restart recovery
Forward 6 candidate factual NLL=1.64015 versus original=1.64063 (-0.00048) on 50 tuning stays. Admission-null insulin candidate passes this subset; intervention accuracy remains untested. Recovered architecture, validation IDs, causal notes and retained objects. No weight changes. Python runs: 59; forwards: 6.

## Forward 7 and Run 60: intervention check
On the same 50 tuning stays, original/current insulin x2=-1.672/-7.810, half=+0.784/+2.697; NE x2=-0.548/-0.311, half=-0.726/-1.036. Candidate improves insulin but remains underpowered and asymmetric; NE requires its own evidence-backed repair. Recovering retained fits before revising. No weight changes. Python runs: 60; forwards: 7.

## Run 61: compact helper recovery
Previous output truncated before helper inventory. Retrieving only helper signatures, recent edit arrays and causal-object names. No weight changes. Python runs: 61; forwards: 7.

## Runs 62–63: balanced insulin candidate
Run 62 verified raw-rate contrast units; old feature_metrics helper failed, so diagnostics were recomputed explicitly. Closed-form precision-weighted correction enforces equal mean insulin x2/half sensitivity, zero mean NE cross-response, and exact admission nullspace. Base slopes=[1.0997098912801242, 0.3794443556031091, 1.7219279116606785, -0.7281142378782448], balanced=[1.0997098912801242, 1.0997098912800993, -1.4210854715202004e-14, -9.947598300641403e-14]; heldout factual RMS 0.369017/0.711591; initial RMS 4.71e-13. Strength relative to candidate58=1.9; installed=True. Not accepted before GPU validation. Python runs: 63; forwards: 7.

## Forward 8 and Run 64: reject stronger balanced edit
Candidate63 factual NLL=1.64769 versus original=1.63461 (+0.01308), failing the 0.005 ceiling. Restored candidate58. Refitted admission-null balanced insulin features with increasing factual penalties using closed-form constrained regressions; metrics={10.0: {'heldout_slot_rms': 0.7115907649546053, 'heldout_mean_rms': 0.7054814849400769, 'contrast_rmse': 1.349986335718104, 'insulin_wrong_sign_fraction': 0.14320388349514562, 'norm': 245.00693258833718, 'admission_rms': 4.19159994287174e-13}, 50.0: {'heldout_slot_rms': 0.6749155594453787, 'heldout_mean_rms': 0.6679650372957756, 'contrast_rmse': 1.5151482877697735, 'insulin_wrong_sign_fraction': 0.14320388349514562, 'norm': 262.8558390688549, 'admission_rms': 4.81965909162671e-13}, 200.0: {'heldout_slot_rms': 0.6711737699454079, 'heldout_mean_rms': 0.6638420644110243, 'contrast_rmse': 1.571476639773249, 'insulin_wrong_sign_fraction': 0.14805825242718446, 'norm': 308.73414450392295, 'admission_rms': 7.232194235484413e-13}, 1000.0: {'heldout_slot_rms': 0.6693575744036792, 'heldout_mean_rms': 0.6619944486135633, 'contrast_rmse': 1.5864924130948734, 'insulin_wrong_sign_fraction': 0.15048543689320387, 'norm': 585.0538002640002, 'admission_rms': 1.4644766711158185e-12}, 10000.0: {'heldout_slot_rms': 0.666948021002266, 'heldout_mean_rms': 0.659749334593866, 'contrast_rmse': 1.58413116330007, 'insulin_wrong_sign_fraction': 0.15048543689320387, 'norm': 1376.6345977824853, 'admission_rms': 3.3734545752299232e-12}}. Examined training NE dose policy against current MAP for identification opportunities. Python runs: 64; forwards: 8.

## Run 65: second restart recovery
Candidate63 was rejected for factual degradation; candidate58 remains restored. The NE dose/MAP table is observational policy evidence, not a causal estimate. Reading architecture, early identification notes, and retained objects before further edits. No weights changed. Python runs: 65; forwards: 8.

## Run 66: compact restart recovery
Architecture and early causal evidence recovered; run65 output truncated before helpers. Retrieving bounded helper and object inventories and early edit-development notes. No weights changed. Python runs: 66; forwards: 8.

## Run 67: isolated strength test
Verified current edit equals candidate58, then installed 1.6 times its original delta. Unlike candidate63, this changes no feature balancing or output direction. Candidate is unaccepted pending 100-stay factual validation, then paired rollouts. Python runs: 67; forwards: 8.

## Forwards 9–10 and Run 68: stronger insulin baseline
Candidate58 at strength 1.6 has paired factual NLL increase 0.00278 on 100 random tuning stays. Rollouts: insulin x2=-12.708, half=+4.335; NE x2=+0.099, half=-1.004. Retain as working baseline, not final acceptance. Recovering NE identification and recurrent-edit helpers. No weights changed. Python runs: 68; forwards: 10.

## Run 69: independent NE identification audit
Cross-fitted nuisance regressions on 160000 training intervals, split by stay. Compared next arterial, cuff and pooled MAP and a five-hour pre-treatment placebo. Estimates: {'map_arterial_next': {'n': 73376, 'stays': 14386, 'beta': 3.2115726486497214, 'cluster_se': 0.181778300318948, 'dose_residual_sd': 0.09469558021224553}, 'map_cuff_next': {'n': 97411, 'stays': 27143, 'beta': 2.5618284990665234, 'cluster_se': 0.36550283143603807, 'dose_residual_sd': 0.06577550333630644}, 'map_any_next': {'n': 148905, 'stays': 29889, 'beta': 3.037226355190392, 'cluster_se': 0.19240452555909313, 'dose_residual_sd': 0.07711821826911186}, 'map_five_hours_before': {'n': 124198, 'stays': 29738, 'beta': -0.31009167788799447, 'cluster_se': 0.18961590965331618, 'dose_residual_sd': 0.07629670568894915}}. Computed schedule-target sensitivity on forward-10 reached hours. No weights changed. Python runs: 69; forwards: 10.

## Run 70: third restart recovery
Recovered architecture, early identification notes and retained helpers. NE next-hour estimates agree near +3 MAP/dose unit, but cumulative targets depend on unidentified persistence; estimate that before editing. Insulin working edit remains candidate58 at strength 1.6. No weights changed. Python runs: 70; forwards: 10.

## Run 71: bounded helper inventory
Run70 output was truncated before the inventory. Retrieved bounded helper/object summaries to reuse the established local reconstruction. No weights changed. Python runs: 71; forwards: 10.

## Run 72: dynamic NE identification setup
Inspected panel, run69 objects and causal helper definitions. Next estimate the innovation response across horizons and jointly account for induced later doses; do not equate observational MAP autocorrelation with causal persistence. No weights changed. Python runs: 72; forwards: 10.

## Run 73: dynamic NE innovation identification
Cross-fitted future MAP and future dose nuisance models on a common t<=16 cohort. Jointly fit one-hour effect and persistence to MAP innovation moments, accounting for induced subsequent doses. Moments={1: {'n': 94452, 'beta': -0.02501344005107875, 'se': 1.04838270976256}, 2: {'n': 97405, 'beta': -1.0135182157036144, 'se': 0.6100605455903194}, 3: {'n': 93462, 'beta': -1.2228509484341887, 'se': 0.6199590540707861}, 4: {'n': 89209, 'beta': -0.6984357074500345, 'se': 0.6356329470532504}, 6: {'n': 81118, 'beta': -1.1122136791348598, 'se': 0.6730170106079933}, 8: {'n': 73092, 'beta': -0.5550111798593667, 'se': 0.7164405256503443}}; dose moments={1: [1.0], 2: [1.0, 0.01914900375353457], 3: [1.0, 0.024263955077150918, 0.02412448431944555], 4: [1.0, 0.021329387098985554, 0.020900864377022942, 0.026311400445344777], 6: [1.0, 0.0295175903312197, 0.028742944681858368, 0.03345312044958435, 0.03467325608332192, 0.03258888550846201], 8: [1.0, 0.012638283124336723, 0.010509397203615156, 0.015868562017731996, 0.016242717650146437, 0.01318490411952469, 0.007581166543398616, 0.0011204137156845314]}; joint fit=(np.float64(1.3082503735627378), np.float64(-0.8973051204084299), np.float64(0.9710000000000001)). These are provisional pending separate arterial/cuff and placebo checks. No weights changed. Python runs: 73; forwards: 10.

## Run 74: reject unreplicated dynamic calculation and audit alignment
Run73 did not reproduce the audited next-hour effect; its beta/persistence are not accepted. Checked source-row versus stay-hour alignment, dose-residual units and rollout labels before any further fitting. No weights changed. Python runs: 74; forwards: 10.

## Run 75: corrected dynamic NE identification
Run74 proved run73 future outcomes/doses were misaligned because s69 has a reset index. Replaced positional indexing with exact (stay_id,t+h) lookup; next-hour records match exactly. Refit aligned dose and outcome nuisances with 240 trees. Dynamic moments={1: {'n': 105503, 'beta': 2.707341790473528, 'se': 0.22928813230515843}, 2: {'n': 105659, 'beta': 4.275426804509154, 'se': 0.2516858427454782}, 3: {'n': 105597, 'beta': 5.286207576109406, 'se': 0.2706583217782913}, 4: {'n': 105472, 'beta': 4.896527605434241, 'se': 0.2905893582442792}, 6: {'n': 105647, 'beta': 4.153821192984478, 'se': 0.318405176478772}, 8: {'n': 105537, 'beta': 2.8446145368009423, 'se': 0.3440323915504443}}; induced-dose moments={1: [1.0], 2: [1.0, 0.7678150937142122], 3: [1.0, 0.7687398101649293, 0.6155828468732708], 4: [1.0, 0.7659682086051842, 0.6118029864532559, 0.4867615334720917], 6: [1.0, 0.7654620988437804, 0.6145966743585412, 0.48591950806038614, 0.39504773569326196, 0.3241656232485549], 8: [1.0, 0.7655747854425458, 0.6134864211621766, 0.48805923923578376, 0.3956880370106514, 0.32349670489057053, 0.26711939581157884, 0.22047323900184745]}; joint fit=(5.057288859576627, 2.9779372252559315, 0.6970000000000001); provisional target={'mean': 4.219367108688972, 'n': 314}. No weights changed. Python runs: 75; forwards: 10.

## Run 76: independent MAP channels and causal uncertainty
Arterial/cuff dynamic checks={'map_arterial': {'estimates': {1: {'n': 51893, 'beta': 2.9387948130277888, 'se': 0.21964029906671806}, 2: {'n': 51940, 'beta': 4.680009584019375, 'se': 0.24861714255824788}, 3: {'n': 51932, 'beta': 5.397717206666027, 'se': 0.2767427349355862}, 4: {'n': 51859, 'beta': 5.2585434808298, 'se': 0.302682560828227}, 6: {'n': 51949, 'beta': 4.3645044465687475, 'se': 0.33745937444494684}, 8: {'n': 51920, 'beta': 3.048246026717235, 'se': 0.367529180249146}}, 'fit': {'loss': 2.555341569475786, 'beta': 3.1670730232094746, 'phi': 0.704}, 'target': {'mean': 4.56014447356125, 'n': 314}}, 'map_cuff': {'estimates': {1: {'n': 69058, 'beta': 2.1020666470455396, 'se': 0.44006518418597923}, 2: {'n': 69251, 'beta': 4.102621742717178, 'se': 0.45135341680031926}, 3: {'n': 69183, 'beta': 5.10635017890205, 'se': 0.47490517092487944}, 4: {'n': 69137, 'beta': 4.626570876278255, 'se': 0.49924869801168564}, 6: {'n': 69196, 'beta': 4.1102847972633265, 'se': 0.537797071089945}, 8: {'n': 69420, 'beta': 2.7429559754483934, 'se': 0.5730563076300162}}, 'fit': {'loss': 3.9035907026925543, 'beta': 2.7222466117192488, 'phi': 0.7090000000000001}, 'target': {'mean': 3.9654392548953075, 'n': 314}}}; pre-treatment placebos={5: {'n': 80743, 'beta': -0.12328763814547097, 'se': 0.23138970019990218}, 8: {'n': 62176, 'beta': 0.20496251801090296, 'se': 0.28579034066169073}}; cluster-bootstrap beta/phi quantiles=[[2.6934940177958335, 0.6509750000000001], [2.989049431534111, 0.6990000000000001], [3.228756742688265, 0.7360250000000003]]; NE x2 target quantiles=[3.779025713488793, 4.240095227778519, 4.730018491810367]. No weights changed. Python runs: 76; forwards: 10.

## Run 77: freeze insulin baseline and recover NE edit machinery
Independent arterial/cuff persistence is about .70; placebo effects are near zero. Working NE x2 target about +4.2 to +4.6, half about -2.1 to -2.3. Saved current insulin-only b_in delta to /work/baseline77.npz before NE surgery. No weights changed. Python runs: 77; forwards: 10.

## Run 78: restart recovery for NE surgery
Baseline77 is preserved. Recovering architecture and constrained-feature objects before a reversible NE edit; no weights changed. Python runs: 78; forwards: 10.

## Run 79: NE feature and direction unit audit
Inspecting constrained factual leakage, dose-target units and local MAP/glucose update sensitivities before choosing a NE correction. No weights changed. Python runs: 79; forwards: 10.

## Run 80: balanced NE detector setup
Existing factual-zero NE feature is strongly asymmetric between doubling and halving. Auditing raw feature units and held-out stay masks before a separately balanced closed-form fit. No weights changed. Python runs: 80; forwards: 10.

## Run 81: independent balanced NE feature fits
Raw-unit held-out slopes of the old NE feature are .11648 (double) and .05022 (half); earlier keyed diagnostics were not these slopes. Fitted closed-form NE detectors with factual penalties, exact admission null constraints, insulin-contrast guards, and three halving weights. Forward-only checked the MAP output direction. No weights changed. Python runs: 81; forwards: 10.

## Run 82: selective NE detector and multi-context MAP direction
Heavy halving weights can reverse doubling sensitivity. Strengthened insulin contrast guards and factual penalties; fitted a broader glucose-neutral MAP output direction using forward differences only. Selected detector=(1.0, 1.0, 300.0), metrics={'cross': 1.0, 'half_w': 1.0, 'anchor': 300.0, 'norm': 1.0216780374234442, 'factual_rms': 0.0033377344221220823, 'cross_rms_2.0': 0.047949565741222555, 'cross_rms_0.5': 0.034422219915327616, 's2.0': 0.05268984731541728, 's0.5': 0.020339962206439764, 'unit_factual': 0.06334682281657412, 'unit_cross': 1.5633331629041907, 'half_ratio': 0.38603190638755563, 'score': 0.3809324982057856}. No weights changed. Python runs: 82; forwards: 10.

## Run 83: reversible insulin+NE trial installed
Added a rank-one NE correction to the preserved insulin baseline. Detector=(1.0, 1.0, 300.0); multi-context MAP direction ridge=.01; strength=56.937, NE delta norm=714.41. Scaling uses record-based NE beta=3.0, not external medical knowledge. Trial is unaccepted pending paired factual and intervention validation. Saved /work/ne_trial83.npz. Python runs: 83; forwards: 10.

## Run 84: NE trial fails factual constraint
Forward11: original NLL 1.60516, edited 1.63855; increase .03339 exceeds .005. Largest deterioration is arterial MAP (.0788), with smaller glucose and cuff losses. Recovering architecture, saved baseline and edit helpers before revision. NE trial83 is unaccepted. Python runs: 84; forwards: 11.

## Run 85: reject NE trial and restore insulin baseline
Forward12 trial responses: insulin x2 -12.512, half +3.326; NE x2 +1.543, half -3.791. NE responses are imbalanced and factual NLL failed by +.03339. Restored original b_in plus baseline77 insulin-only delta. Recovering prior validation and local forward helpers for a more selective NE edit. Python runs: 85; forwards: 12.

## Run 86: NE interference and helper recovery
Baseline forward10 insulin shifts -12.708/+4.335, NE +.099/-1.004. NE trial reduced insulin-half response by ~1.0 and failed factual fit. Insulin-only baseline remains installed. Recovering local forward and constrained regression objects to improve detector selectivity and doubling/halving balance. Python runs: 86; forwards: 12.

## Run 87: detector feasibility audit
Inspecting contrast targets, masks, raw features and local coda helpers before closed-form selective refitting. No new weight changes; baseline77 remains installed. Python runs: 87; forwards: 12.

## Run 88: NE signal/leakage frontier
Comparing retained NE feature fits, checking basis and raw-dose units, and auditing paired factual mean/NLL shifts. Baseline77 remains installed. Python runs: 88; forwards: 12.

## Run 89: consider nonlinear posterior feature route
Retained b_in linear detectors have substantial insulin cross-response even with stronger guards. NE trial factual MAP mean shifts vary by ~1.7, so a constant debias is inadequate. Inspecting posterior hidden-feature capacity and prior local-edit experiments. No weights changed; baseline77 remains installed. Python runs: 89; forwards: 12.

## Run 90: restart recovery and selective-edit audit
Insulin-only baseline77 remains installed. Recovering architecture, prior insulin validation, and retained NE/core helpers before a different NE correction. No weights changed. Python runs: 90; forwards: 12.

## Run 91: compact helper recovery
Architecture recovered; run90 output truncated before helpers. Inspecting retained NE fits and validated insulin edit without another large architecture dump. No weights changed. Python runs: 91; forwards: 12.

## Run 92: nonlinear prelude feature route
Considering a closed-form detector on the prelude FFN's 768 hidden features rather than the 192-dimensional normalized prelude. Need held-out factual-null, admission-null, drug-specific and balanced contrast checks before a reversible trial. No weights changed. Python runs: 92; forwards: 12.

## Run 93: nonlinear factual-centered NE detector fits
Extracted 768 prelude FFN hidden features using GPU posterior means for factual bundles. Closed-form fits include a bias, exact initial admission nullspace, factual-zero penalties and insulin-contrast guards. Held-out diagnostics printed; no weights changed. Python runs: 93; forwards: 12.

## Run 94: correct nonlinear feature alignment
Run93 used inconsistent observation/action vectors on drug bundles; its feature fits are rejected. Verified factor-one posterior before rebuilding factual and edited drug features consistently. Corrected held-out fits printed. No weights changed. Python runs: 94; forwards: 12.

## Run 95: corrected recurrent feature audit
Factor-one drug posterior is exact. Rebuilt normalized prelude features with consistent drug observation/action encodings and true factual posterior means. Audited old NE detector leakage and refitted constrained 192-feature detectors before any new weight edit. Python runs: 95; forwards: 12.

## Run 96: factual dose-innovation error audit
Corrected old NE detector has materially greater factual leakage and halving than doubling sensitivity. Tested whether original next-hour arterial MAP residuals retain a cross-fitted dose-innovation signal, using only tuning records and original predictions. This is a diagnostic, not an independent causal estimate. No weights changed. Audit beta=nan, cluster SE=nan, n=0. Python runs: 96; forwards: 12.

## Run 97: recover targets and fix empty residual audit
Run96 matched zero future MAP readings; its NaNs are not evidence of a null effect. Inspecting record/prediction keys, causal estimates and model interfaces. No weights changed; baseline77 remains installed. Python runs: 97; forwards: 12.

## Run 98: compact evidence and feature recovery
Architecture and record schema recovered. Fractional infusion hours may explain the empty exact-hour residual match; recovering prior causal targets and retained feature helpers. No weights changed. Python runs: 98; forwards: 12.

## Run 99: corrected timing and compact feature recovery
Next recorded arterial matching found 947/991 NE records within two hours, avoiding fractional-hour equality. Recovering corrected features and record-derived NE dynamics. No weights changed. Python runs: 99; forwards: 12.

## Run 100 results and Run 101: posterior-feature alternative
Corrected original residual audit: n=501, beta=0.9844, cluster SE=1.0352; inconclusive diagnostic. Corrected prelude-hidden and b_in detectors have factual leakage around .37-.69 raw-dose units per unit doubling sensitivity; not yet safe to scale. Run100 completed fits but failed during its final inventory. Inspecting posterior features as a different closed-form edit route. No weights changed. Python runs: 101; forwards: 12.

## Run 102: posterior detector frontier and local reconstruction
Factor-one posterior inputs are exact. Fitted initial-admission-null posterior detectors with factual and insulin guards, evaluated by held-out stays. Checked local recurrent reconstruction against original GPU states. No weights changed. Python runs: 102; forwards: 12.

## Run 103: modest corrected balanced NE trial
Posterior route rejected as no more selective. Local original reconstruction out RMSE=.0507, so GPU tests remain decisive. Installed baseline77 plus a corrected balanced b_in NE rank-one trial, beta correction=1.0, metrics={'half_w': 3.0, 's2': 0.000950745308085738, 's_half': 0.00097385013930297, 'half_ratio': 1.0243018093497112, 'fact_per_x2': 0.37156234535742527}, NE delta norm=1104.404. Trial unaccepted pending factual and paired rollout tests. Saved ne_trial103.npz. Python runs: 103; forwards: 12.

## Run 104: reject corrected NE trial
Forward13 factual original=1.60883, trial=1.62605; +.01722 violates .005. Trial103 rejected. Recovering baseline archive and prior causal evidence. Python runs: 104; forwards: 13.

## Run 107: reduced corrected NE trial
Run106 inventory errored on a non-array; weights were unchanged. Verified trial103 composition (total error 500.611, NE-only error 0). Installed baseline77 plus .35 times trial103 NE delta; unaccepted pending paired factual and rollout tests. Python runs: 107; forwards: 13.

## Run 108: reduced NE passes, but insufficient causal correction
Forward14 NLL increase=.00402. Forward15 shifts: insulin -12.555/+4.182, NE +.587/-1.282. Reduced NE improves both signs but is far below record-derived doubling target. Investigating actual per-slot b_in perturbations and MAP-selective output directions before scaling. Python runs: 108; forwards: 15.

## Run 110: restart recovery before redesign
Run109 stopped during inspection; no new weights were installed. Recovering architecture, record-derived effects, saved edit composition, and local helpers. Reduced NE correction remains installed; its tested factual NLL increase was .00402, leaving little safety margin. Python runs: 110; forwards: 15.

## Run 111: focused recovery
Early records-based estimates support negative insulin and positive NE effects, but later estimates and edit calibration must be recovered before revision. No weights changed. Python runs: 111; forwards: 15.

## Run 112: recover direction construction
Run111 output was repeated; it supplied no new experiment. Inspecting detector/output-direction construction and forward-only helpers for a MAP-selective correction. No weights changed. Python runs: 112; forwards: 15.

## Runs 113–114: robust perturbation audit
Run113 stopped because E_fact95 is an ndarray; it changed no weights. Run114 repeats the diagnostic with explicit array/tensor conversion. Recovering factual dose-innovation targets before choosing a new detector. Python runs: 114; forwards: 15.

## Run 115: factual innovation-targeted NE detectors
Full trial activates non-NE factual bundles and has large hnew perturbations. Fitted closed-form prelude probes targeting record dose innovations on active NE bundles and zero elsewhere, with synthetic dose contrasts, insulin guards, and admission-null constraints. Held-out stay diagnostics printed. No weights changed. Python runs: 115; forwards: 15.

## Run 116: reject innovation-probe route; recover causal targets and dose pathway
Run115 innovation probes have poor held-out innovation fits and uneven doubling/halving gains; none installed. Recovering original architecture, record-derived target estimates, accepted insulin edit and retained helpers. Reduced NE correction remains installed. Python runs: 116; forwards: 15.

## Run 117: compact calibration/helper recovery
Run116 architecture confirms a shared recurrent dose pathway; output was truncated before helper inventory. Retrieving only calibration/validation notes and recent objects to avoid repeated large dumps. No weights changed. Python runs: 117; forwards: 15.

## Run 118: output-constrained edit feasibility
Recovered NE schedule target +4.2..4.6 / -2.1..2.3. Scalar probe edits trade causal gain for factual leakage. Inspecting local coda/readout helpers and retained states for a closed-form output-constrained, multi-direction edit. No weights changed. Python runs: 118; forwards: 15.

## Run 119: inspect output-constrained inputs
Preparing a closed-form correction constrained on actual reading outputs, rather than scalar probe leakage alone. Inspecting retained local states, reading-head arrays and metadata. No weights changed. Python runs: 119; forwards: 15.

## Run 120: per-slot, multi-direction output sensitivities
Computed central finite-difference coda/reading sensitivities for 14 recurrent directions in each of four slots on retained local contexts; no gradients and no weight changes. Preparing a larger output-constrained closed-form regression if sensitivities provide useful selectivity. Python runs: 120; forwards: 15.

## Run 121: all-bundle reconstruction and distributional readout
Slot sensitivities are nearly identical, supporting a slot-mean structured regression. Reconstructed all 4356 bundles; out RMSE=0.082293. Implemented generic mean/logit/probability readout and checked against retained decoder. No weights changed. Python runs: 121; forwards: 15.

## Run 122: reject incorrect decoder and isolate mismatch
Run121 local out RMSE=.0823, but generic decoder differed from trusted helper by 68.3 units; no fit or weight edit made. Auditing exact helper computation and register offsets before any distributional constraints. Python runs: 122; forwards: 15.

## Run 123: recover exact decoder semantics
Register offsets alone did not fix the decoder. Reading architecture and complete trusted helper before using any local output constraints. No weights changed; reduced NE edit remains installed. Python runs: 123; forwards: 15.

## Run 124: compact decoder comparison
Run123 output was truncated after architecture; no new numerical evidence. Comparing complete decoder bytecode and captured defaults without large array reprs. No weights changed. Python runs: 124; forwards: 15.

## Run 125: correct reading query type
Found generic decoder type_emb[0] versus trusted type_emb[1]. Implemented type-one distributional readout and checked identical-input equivalence. Inspecting retained GPU targets and all-bundle states before fitting. No weights changed. Python runs: 125; forwards: 15.

## Run 126: validate type-corrected readout
Type-one generic decoder exactly matches trusted helper. Comparing GPU and reconstructed states by outcome and recovering sensitivity/causal calibration inputs. No weights changed. Python runs: 126; forwards: 15.

## Run 127: audit local approximation and recover contrasts
Corrected original-state decoder RMSE=3.00 overall, 1.57 arterial; reconstructed-state glucose errors remain large. Auditing alignment and recovering exact corrected prelude contrasts before fitting output-constrained NE correction. Current weights unchanged. Python runs: 127; forwards: 15.

## Run 128: hidden-state and edit-composition audit
Reading source alignment is exact. GPU/local h RMSE=6.11; output-normalization hid a major reconstruction discrepancy. Corrected archive interpretation: ne_trial103 delta is NE-only, not baseline+NE; installed weights were never changed by this bookkeeping error. Inspecting helper weight sources and contrast indexing. Python runs: 128; forwards: 15.

## Run 130: recover after attention reshape failure
Run129 failed before validation and changed no weights. Inspecting query/key dimensions and helper definitions; requiring hidden-state agreement before relying on local output constraints. Python runs: 130; forwards: 15.

## Run 131: compact attention/input inventory
Run130 output truncated during bytecode; no numerical test or edit. Recovering compact input shapes, core helper metadata and dose-edit semantics to fix the reconstruction. Python runs: 131; forwards: 15.

## Run 132: identify batch-selection error
Run131 showed query batch 96 but memory batch 4, explaining Run129's reshape failure. Recovering memory helper indexing for a correctly batched hidden-state validation. No weights changed. Python runs: 132; forwards: 15.

## Run 133: correctly batched recurrent reconstruction
Rebuilt memory with explicit row selection and swept attention age semantics/head counts against original GPU states. No weights changed.
 heads  age_divisor       age_mode  include_current    h_rmse  out_rmse
     4          1.0 projected_both             True  5.766405  0.081917
     4          1.0 projected_both            False  9.789918  0.126329
     4          1.0  projected_key             True 11.973384  0.158994
     4       3600.0 projected_both             True 12.684074  0.164497
     4       3600.0 projected_both            False 12.799352  0.165585
Python runs: 133; forwards: 15.

## Run 134: bypass uncertain recurrent reconstruction
Run133 best raw-state RMSE remained 5.77. Recovered pre-coda inputs from original GPU h using forward-only fixed-point inversion; no gradients or weight changes. Replay h RMSE=1.9144582e-06, out RMSE=8.1896609e-08. This supplies an exact local anchor for immediate b_in perturbations, not a validated full rollout reconstruction. Python runs: 134; forwards: 15.

## Run 135: recover output-regression inputs
Coda inversion is numerically exact (h RMSE 1.91e-6), avoiding the unvalidated recurrent reconstruction. Recovering reading queries, corrected counterfactual features and record-derived targets for a distribution-constrained closed-form edit. No weights changed. Python runs: 135; forwards: 15.

## Run 137: safe contrast alignment inventory
Run136 completed anchored readout and all 14 sensitivity directions (decoder agreement exact); only its metadata inventory failed on a class descriptor. Retained sensitivities unchanged. Recovering contrast pairing and bundle indices safely. No weights changed. Python runs: 137; forwards: 15.

## Run 138: recover calibration and exact-anchor sensitivities
Run137 confirmed drug-contrast alignment but supplied no new intervention validation. Recovering compact effect estimates and exact-coda sensitivity objects before fitting a reading-output-constrained correction. No weights changed. Python runs: 138; forwards: 15.

## Run 139: bounded calibration recovery
Run138 output was truncated before newer validation and sensitivity metadata. Retrieving only later validation summaries and exact-anchor object shapes. No weights changed. Python runs: 139; forwards: 15.

## Run 140: output-constrained weight-interface audit
Recovered current readouts insulin -12.555/+4.182, NE +.587/-1.282, factual NLL increase .00402. Insulin-only baseline increase .00278. Record-supported NE schedule target +4.2..4.6/-2.1..2.3. Checking exact coda sensitivity interface and admission-null feature basis before a multi-direction closed-form correction. No weights changed. Python runs: 140; forwards: 15.

## Run 141: verify output-regression alignment
Audited admission-null features, corrected dose contrasts, raw NE units, orthonormalized output directions and the exact b_in-to-coda linear map. Preparing stay-held-out distributional and dose-response constraints. No weights changed. Python runs: 141; forwards: 15.

## Run 142: resolve projection and recurrent-scale discrepancies
Historical basis matches: [(0.6591837780527738, 'Binput141'), (0.6591837780527738, 'basis100')]. Recurrent scale-map errors: {0.0: 0.016722026857412767, 0.0001: 0.01669264353017717, 0.001: 0.01645320152044087, 0.01: 0.016746425362764362}. Unchanged-dose contrast maxima: {('insulin', 2.0): 0.0, ('insulin', 0.5): 0.0, ('norepinephrine', 2.0): 2.384185791015625e-07, ('norepinephrine', 0.5): 2.384185791015625e-07}. Fresh initial-prelude null rank=4; ready=False. No weights changed. Python runs: 142; forwards: 15.

## Run 143: fresh weight-mapped output sensitivities
Historical XF95 and prebasis136 discrepancies were not explained; they are not used in the new regression. Recomputed 14 central finite-difference reading sensitivities using original softplus(delta_in), c_out, exact GPU coda anchors, and fresh admission-null features. No gradients and no weights changed. Python runs: 143; forwards: 15.

## Run 144: closed-form multi-direction NE correction frontier
Fit 14-output admission-null regressions with factual Fisher-distribution penalties and insulin/glucose guards, using stay-held-out local validation. Increment targets are NE beta 3.0 for doubling and 1.6 for halving, reflecting record-derived schedule effects minus the insulin-only baseline's existing responses. Local frontier: {"0.3": {"norm": 393.110627084638, "heldout_KL_linear": 0.0036458427863599575, "heldout_mean_rms": {"map_arterial": 0.5150886924851277, "map_cuff": 0.6659633958782601, "glucose_finger": 0.6670685079851056, "glucose_lab": 0.6185929507439407}, "NE_2.0_slope": 1.8788694042602, "NE_0.5_slope": 0.681538400195289}, "1.0": {"norm": 400.6469834745396, "heldout_KL_linear": 0.0021595751709915967, "heldout_mean_rms": {"map_arterial": 0.42534579535691136, "map_cuff": 0.5104261884933562, "glucose_finger": 0.44675928853128655, "glucose_lab": 0.39942645733123244}, "NE_2.0_slope": 1.8215252401670186, "NE_0.5_slope": 0.6631651527006814}, "3.0": {"norm": 416.3004391787182, "heldout_KL_linear": 0.0014984963280380347, "heldout_mean_rms": {"map_arterial": 0.3663539325051326, "map_cuff": 0.4275398678065313, "glucose_finger": 0.3123446398400817, "glucose_lab": 0.3115476462227195}, "NE_2.0_slope": 1.734090526029308, "NE_0.5_slope": 0.6291098592700494}, "10.0": {"norm": 448.0096910757974, "heldout_KL_linear": 0.000992679519734656, "heldout_mean_rms": {"map_arterial": 0.2983948761299037, "map_cuff": 0.35560194375192394, "glucose_finger": 0.23175761052245827, "glucose_lab": 0.268588589587403}, "NE_2.0_slope": 1.5639196438540544, "NE_0.5_slope": 0.5589914665712908}, "30.0": {"norm": 483.5945024053507, "heldout_KL_linear": 0.0006166961611151329, "heldout_mean_rms": {"map_arterial": 0.22825575863993616, "map_cuff": 0.2932282040446467, "glucose_finger": 0.18151368008106197, "glucose_lab": 0.22230568957495833}, "NE_2.0_slope": 1.3143725955672914, "NE_0.5_slope": 0.45828919181978955}}. No weights changed; local metrics are not rollout validation. Python runs: 144; forwards: 15.

## Run 145: restart recovery
Recovered architecture, early record-evidence notes, and live regression objects before any new candidate installation. Run144 local frontier is not a rollout or factual-NLL validation. No weights changed. Python runs: 145; forwards: 15.

## Run 146: bounded candidate recovery
Retrieved regression object names, saved-candidate tensor layouts, and selected earlier validation notes. No weights changed. Python runs: 146; forwards: 15.

## Run 147: candidate weight-map verification
Audited admission-null basis candidates, output-direction orthogonality, readout helper interface, and current baseline/NE composition. No weights changed. Python runs: 147; forwards: 15.

## Run 148: verified multi-direction NE trial
Mapping checks: {'sensitivity_map_max_error': 0.0, 'bundle_feature_max_error': 0.0, 'reading_feature_max_error': 0.0, 'admission_null_max_error': 2.5045925859745447e-14, 'weight_direction_gram_error': 7.742604430157619e-15, 'candidate_norm': 448.00969107579783, 'reported_norm': 448.0096910757974}. Installed=True. If installed, composition is original b_in + baseline77 + Bdir141 @ fits144[10] @ Binput142.T, replacing—not adding to—the old NE trial. Candidate remains unaccepted until GPU factual and paired-rollout tests. Python runs: 148; forwards: 15.

## Forward 16 / Run 149: reject multi-direction mean-feature trial
Candidate148 factual NLL=1.72932 vs original=1.63461 (+0.09471), decisively failing the +0.005 limit. Restored the previous feasible edit, original + baseline77 + .35*ne_trial103 NE delta. Examined per-slot pre-coda perturbations because uniform-slot/mean-feature local penalties are insufficient. Python runs: 149; forwards: 16.

## Run 150: constrain factual state changes
Run149 showed pre-coda slot RMS=6.145 and bundle-mean RMS=6.094; large decoder-insensitive changes, not mainly slot cancellation. Recovered the prior normal equations and fit a closed-form frontier with an additional per-slot pre-coda state penalty. Frontier: {0.003: {'weight_norm': 428.91370664119006, 'heldout_state_rms': 0.6686766326230452, 'heldout_local_KL': 0.0008542789624243658, "('insulin', 2.0)_guard_rms": 0.015458623300037305, "('insulin', 0.5)_guard_rms": 0.011697419769045824, "('norepinephrine', 2.0)_slope": 1.70071021913493, "('norepinephrine', 0.5)_slope": 0.7711159807842108}, 0.01: {'weight_norm': 423.18641215140457, 'heldout_state_rms': 0.4668712257297878, 'heldout_local_KL': 0.0008241641121953357, "('insulin', 2.0)_guard_rms": 0.015285532823205042, "('insulin', 0.5)_guard_rms": 0.011489992188461785, "('norepinephrine', 2.0)_slope": 1.654820834142511, "('norepinephrine', 0.5)_slope": 0.758067167721415}, 0.03: {'weight_norm': 418.62879545736405, 'heldout_state_rms': 0.3264494337344484, 'heldout_local_KL': 0.0008040053892774829, "('insulin', 2.0)_guard_rms": 0.015315308888588784, "('insulin', 0.5)_guard_rms": 0.011292338296060836, "('norepinephrine', 2.0)_slope": 1.5945855467149777, "('norepinephrine', 0.5)_slope": 0.7391673575480172}, 0.1: {'weight_norm': 411.65730560977664, 'heldout_state_rms': 0.22832601755075485, 'heldout_local_KL': 0.0007625981011846209, "('insulin', 2.0)_guard_rms": 0.015505659082560928, "('insulin', 0.5)_guard_rms": 0.011024688799199274, "('norepinephrine', 2.0)_slope": 1.501458251334809, "('norepinephrine', 0.5)_slope": 0.7062199156830888}, 0.3: {'weight_norm': 401.20178499282537, 'heldout_state_rms': 0.17384379537979164, 'heldout_local_KL': 0.0006663793125684548, "('insulin', 2.0)_guard_rms": 0.015585903083009375, "('insulin', 0.5)_guard_rms": 0.010654924920337393, "('norepinephrine', 2.0)_slope": 1.367982021346044, "('norepinephrine', 0.5)_slope": 0.6554258462857017}, 1.0: {'weight_norm': 365.6340651223584, 'heldout_state_rms': 0.12626966402947493, 'heldout_local_KL': 0.00047920815430987186, "('insulin', 2.0)_guard_rms": 0.014960330759871026, "('insulin', 0.5)_guard_rms": 0.009788718137169288, "('norepinephrine', 2.0)_slope": 1.122494639273152, "('norepinephrine', 0.5)_slope": 0.5582436692015808}, 3.0: {'weight_norm': 282.82599706058056, 'heldout_state_rms': 0.08421291565285338, 'heldout_local_KL': 0.00026462005348530806, "('insulin', 2.0)_guard_rms": 0.012617830729508587, "('insulin', 0.5)_guard_rms": 0.008031805591134377, "('norepinephrine', 2.0)_slope": 0.794259372118808, "('norepinephrine', 0.5)_slope": 0.42411980317044834}}. No weights changed. Python runs: 150; forwards: 16.

## Run 151: state-penalized NE candidate installed
Installed original b_in + baseline77 + Bdir141 @ fits150[.1] @ Binput142.T. Anchored held-out incremental KL=0.0007436; this is a local diagnostic, not full factual NLL. GPU factual and dose validation pending. Python runs: 151; forwards: 16.

## Run 152: restart recovery before GPU validation
Recovered architecture, record definitions, earlier effect evidence, and factual-validation stay IDs. Candidate151 remains unaccepted; no weights changed. Python runs: 152; forwards: 16.

## Run 153: compact validation and causal-target recovery
Retrieved exact fixed tuning IDs and selected effect/baseline evidence after prior output truncation. No weights changed. Python runs: 153; forwards: 16.

## Forward 17–18 / Run 154: reject gamma=.1; test stronger state penalty
Candidate151 factual NLL=1.64568 vs original=1.63461 (+.01107), failing the constraint. Its rollout shifts were insulin -11.961/+4.922 and NE +1.893/-1.633. Reconstructed baseline77 exactly from saved archive and installed original + baseline77 + Bdir141 @ fits150[1.0] @ Binput142.T. This is an unaccepted gamma=1.0 trial. No gradients or weight optimizer. Python runs: 154; forwards: 18.

## Forward 19 / Run 155: state penalty fails; direct shrinkage trial
Gamma=1 candidate factual NLL=1.64587 (+.01126), rejected. Installed original + verified baseline77 + .4*NE_candidate151_delta, saved as ne_candidate155.npz. Recovery of historical factual results and insulin baseline evidence accompanies this reversible, unaccepted trial. Python runs: 155; forwards: 19.

## Forward 20 / Run 156: isolate insulin baseline factual cost
Candidate155 NLL=1.64113 vs original=1.63461 (+.00652) on first 100 tuning stays. Installed insulin-only baseline77 to measure its contribution on this exact sample. Recovered decoder interfaces and original-state archive layouts for a possible algebraic calibration, not a gradient-based weight fit. Python runs: 156; forwards: 20.

## Run 157: restart recovery before insulin-only validation
Read architecture and record definitions; recovered early causal evidence and fixed validation IDs. Insulin-only baseline77 remains installed; no new validation or weight changes. Python runs: 157; forwards: 20.

## Run 158: bounded recovery
Recovered factual-test stay IDs and selected later causal/baseline evidence after truncated output. No weights changed. Python runs: 158; forwards: 20.

## Forward 21 / Run 159: insulin-only cost and calibration feasibility
Insulin-only baseline77 factual NLL=1.63925 vs original=1.63461 (+.00464) on first 100 tuning stays: passes narrowly. Inspecting decoder interfaces for a small record-based calibration to recover factual margin without a model-weight optimizer. No weights changed. Python runs: 159; forwards: 21.

## Run 160: decoder calibration diagnostics
Insulin-only factual cost is concentrated in glucose. Checked original-state distributional decoder interface and recovered training IDs, vocabulary structure and insulin target evidence. No weights changed. Python runs: 160; forwards: 21.

## Forward 22 / Run 161: training calibration inputs
Collected paired original/insulin-baseline states on 100 training stays. Audited decoder/register computation and filtered exact matching records before calibration. No weights changed. Python runs: 161; forwards: 22.

## Run 162: register and chronological alignment audit
Rebuilt training reading metadata; checked prior-state selection and same-code registers against existing tuning metadata before any calibration. No weights changed. Python runs: 162; forwards: 22.

## Run 163: restart recovery before factual calibration
Read architecture first, recovered early record evidence and selected validation notes, and inventoried decoder/calibration objects. No weights changed. Register alignment is supported; bin-boundary convention still needs resolution. Python runs: 163; forwards: 22.

## Run 164: compact decoder/calibration recovery
Recovered recent calibration objects, decoder tensor names, and helper signatures with bounded output after run163 truncation. No weights changed. Next gate is local/GPU decoder agreement before calibration. Python runs: 164; forwards: 22.

## Run 165: probability decoder recovery
Insulin-only NLL margin is .00036 nats on the first tuning sample. Inspected probability/readout helpers and training-state metadata for a record-based, algebraically installed calibration. No weights changed. Python runs: 165; forwards: 22.

## Run 166: inspect exact local decoder implementation
Recovered decoder bytecode/defaults and register/static inputs before constructing training probabilities. Glucose means shift only modestly; calibration must address probability distributions, not only mean bias. No weights changed. Python runs: 166; forwards: 22.

## Run 167: decoder static-input audit
Built fresh static inputs with register columns 0=has_last, 3=log1p(age)/10, 6=last_z and left-edge bins. Compared against retained decoder inputs and checked deterministic local/GPU NLL and means. No weights changed. Python runs: 167; forwards: 22.

## Run 168: correct calibration row ordering
Run167 compared metadata in raw record order against GPU/head order; its large static errors and NLL were invalid. Reindexed by stay/hour/event, repeated exact static/transition checks, and built paired training inputs. Aligned local NLL=1.667767; GPU NLL=1.634615. No weights changed. Python runs: 168; forwards: 22.

## Run 169: isolate remaining decoder offset
Aligned transitions match exactly and numeric registers match to 1.2e-7, but static input RMS mismatch remains .3407. Tested constant and register-linear residual explanations and inspected trusted query code. No weights changed. Python runs: 169; forwards: 22.

## Run 170: targeted decoder recovery
Run169's static residual is explained to numerical precision by register features. Inspecting architecture and retained helpers before correcting assembly; no weights changed. Python runs: 170; forwards: 22.

## Run 171: locate register projection discrepancy
Comparing the exactly register-linear static residual with learned numeric projection columns and inspecting builder indexing. No weights changed. Python runs: 171; forwards: 22.

## Run 172: distinguish missing from duplicated numeric projection
Residual equals last_numeric.weight times the three populated register features exactly. Auditing retained reference and probability helpers against GPU output before choosing assembly. No weights changed. Python runs: 172; forwards: 22.

## Run 173: bounded reference audit
Run172 output truncated due tensor-valued function defaults. Repeating only decisive array comparisons and compact head/readout code. No weights changed. Python runs: 173; forwards: 22.

## Run 174: GPU arbitration of register assembly
Rebuilt static equals base162; retained static_all122 contains an additional numeric projection. Comparing both original-state decoders directly with GPU factual means and NLLs. No weights changed. Python runs: 174; forwards: 22.

## Run 175: corrected local decoder and paired training distributions
One numeric projection is correct by GPU agreement. Proper log(dt)/10 query gives deterministic original NLL=1.62508 vs GPU=1.63461 and mean RMSE=1.565; duplicate projection is worse. Built common-prior-draw original/insulin-baseline training probabilities for calibration diagnostics. No weights changed. Python runs: 175; forwards: 22.

## Run 176: cross-validated factual-distribution offsets
Corrected paired training MC NLL: original=1.64055, insulin baseline=1.64332 (+.00276). Testing closed-form code/last-bin logit-residual averages with shrinkage and held-out stays, restricted to glucose codes. No weights changed. Python runs: 176; forwards: 22.

## Run 177: do not install weak calibration; audit measured candidates
Code/last-bin distillation offsets slightly reduce KL but held-out NLL does not consistently improve. No decoder edit installed. Reviewing on-disk factual and rollout results to select the next targeted experiment. Python runs: 177; forwards: 22.

## Run 178: recover causal targets and accepted baseline
Insulin-only baseline77 remains installed. No new weights changed. Recovering architecture, early record-based causal evidence, and exact saved edit before a smaller factual-budgeted joint repair. Python runs: 178; forwards: 22.

## Run 179: bounded baseline and causal recovery
Run178 output truncated before saved-candidate details. Recovering only decisive target/edit summaries and archive keys; no weights changed. Python runs: 179; forwards: 22.

## Run 180: reached-hour audit and insulin construction recovery
Recovered target scales and baseline history; recomputed dose summaries on reached hours, including stay-weighted aggregation. No weights changed. Python runs: 180; forwards: 22.

## Run 181: prepare state-conditioned factual distillation
Reached-hour audit confirms baseline insulin responses around -12.7/+4.3. Inspecting paired training decoder inputs for an auxiliary closed-form ridge correction to the final value-head matrix; no gradients or weight optimiser, and no edit installed yet. Python runs: 181; forwards: 22.

## Run 182: cross-validated hidden-feature factual distillation
Hidden/logit max identity error=7.50723e-06. Five-fold by-stay auxiliary ridge results=[{'alpha': 10.0, 'NLL_change': 0.00157151176752628, 'KL_before': 0.0043922488448866195, 'KL_after': 0.003329595885940907, 'correction_RMS': 0.1366230347878041, 'glucose_NLL_change': 0.0008177769995274611}, {'alpha': 100.0, 'NLL_change': 0.001330771916144981, 'KL_before': 0.0043922488448866195, 'KL_after': 0.0033394540145260092, 'correction_RMS': 0.12052933442618059, 'glucose_NLL_change': 0.000641442823491322}, {'alpha': 1000.0, 'NLL_change': 0.0010053267423189734, 'KL_before': 0.0043922488448866195, 'KL_after': 0.0034964333143941203, 'correction_RMS': 0.09479743179856669, 'glucose_NLL_change': 0.00014516387002541873}, {'alpha': 10000.0, 'NLL_change': 0.0003472309006365475, 'KL_before': 0.0043922488448866195, 'KL_after': 0.0037697548187539238, 'correction_RMS': 0.0660786215088665, 'glucose_NLL_change': -0.0004477830297046002}]. No model-weight optimiser or gradients; no weights installed. Python runs: 182; forwards: 22.

## Run 183: reject hidden-head distillation; inspect posterior inference pathway
Hidden-feature distillation reduced KL but all tested penalties increased stay-held-out NLL; no decoder edit installed. Inspecting whether a record-predictable action component can replace dose-dependent posterior inference while leaving the direct action pathway intact. No weights changed. Python runs: 183; forwards: 22.

## Run 184: baseline-state detector robustness hypothesis
Earlier latent sensitivity was weak, so no posterior edit attempted. Inspecting exact action/world encoders to rebuild prelude detector features on paired original and insulin-baseline training states from forward22. Hypothesis: original-state fits leak on the repaired-state distribution. No weights changed. Python runs: 184; forwards: 22.

## Run 185: recover decision evidence after restart
Read architecture first and recovered compact causal/validation evidence. Current insulin baseline retained; no weights changed. Next decision is a factual-budgeted correction, not an unvalidated detector refit. Python runs: 185; forwards: 22.

## Run 186: bounded validation and archive recovery
Run185 output was truncated; retrieved only key experiment summaries, saved-edit array schemas, and forward-output schemas. No weights changed. Python runs: 186; forwards: 22.

## Run 187: compare measured factual and intervention trade-offs
Recovered forward-result tables and retained NE edit bases. No weights changed. Hypothesis to test: repaired-state anchoring can reduce NE factual leakage relative to original-state fits. Python runs: 187; forwards: 22.

## Runs 188–189: complete effect recovery and safe tensor conversion
Baseline rollout on second tuning sample: insulin -12.708/+4.335; NE +0.099/-1.004. Candidate151 NE +1.893/-1.633 but factual cost .011068 on first sample; reduced107 NE +.587/-1.282 with cost .004018 on second. Independent arterial/cuff dynamic evidence supports NE beta near3, persistence .70. Run188 stopped on Torch/NumPy max conversion; no weights changed. Python runs: 189; forwards: 22.

## Run 190: recover training bundle reconstruction conventions
Installed b_in matches baseline77 exactly. Loaded full records and paired train states for repaired-state detector anchoring. No weights changed. Python runs: 190; forwards: 22.

## Run 191: reconstruct paired training preludes and test anchor drift
Built atoms with audited END convention and checked posterior means against paired GPU states. Measured old NE edit activation on original versus insulin-repaired factual states. No weights changed. Python runs: 191; forwards: 22.

## Run 192: reject anchor-drift explanation; build dose contrasts
Old NE factual state RMS is 1.318 original vs1.321 insulin baseline: drift hypothesis refuted. Built training NE/insulin dose contrasts using recorded factual z plus recomputed posterior differences. Remaining posterior reconstruction offset is explicitly not treated as verified absolute inference. No weights changed. Python runs: 192; forwards: 22.

## Run 193: recover exact contrast-edit construction
Run192 refuted repaired-state anchor drift. Reading architecture and recovering saved NE constructions and contrast variables before a factual-null, dose-responsive auxiliary fit. Posterior offsets remain unverified; no weights changed. Python runs: 193; forwards: 22.

## Run 194: bounded construction recovery
Run193 output truncated before recent variables and archive schemas. Recovering those compactly plus early causal evidence; no weights changed. Python runs: 194; forwards: 22.

## Runs 195–196: archive lookup recovery
Run195 confirmed paired train contrasts but stopped on a missing archive key. No weights changed. Located saved direction schemas and recovered prior NE construction notes. Python runs: 196; forwards: 22.

## Run 197: stay-held-out factual-null NE detector frontier
Admission-null rank=180. Fitted auxiliary linear regressions on paired training dose contrasts, with factual and insulin guards; five-fold validation by stay. No model weights changed. Python runs: 197; forwards: 22.
[{'fact_RMS': 2.1403074273748737, 'NE2_slope': 0.96632225567212, 'NE2_RMS': 1.3063971149349758, 'NEhalf_slope': 2.5242534426225802, 'NEhalf_RMS': 1.479528895867999, 'I2_slope': 0.34898270863077246, 'I2_RMS': 1.2543001826230467, 'Ihalf_slope': 0.4914400753544606, 'Ihalf_RMS': 1.0228269233198362, 'factual_weight': 3.0, 'insulin_weight': 1.0}, {'fact_RMS': 2.142704935396895, 'NE2_slope': 0.969425055656918, 'NE2_RMS': 1.326329819665794, 'NEhalf_slope': 2.5533987867936974, 'NEhalf_RMS': 1.499706275033742, 'I2_slope': 0.3547805346678141, 'I2_RMS': 1.2757734399219025, 'Ihalf_slope': 0.4929941156719938, 'Ihalf_RMS': 1.0420193217986302, 'factual_weight': 30.0, 'insulin_weight': 10.0}, {'fact_RMS': 2.1467434597951174, 'NE2_slope': 0.9695337566753592, 'NE2_RMS': 1.338785566204939, 'NEhalf_slope': 2.569272752401999, 'NEhalf_RMS': 1.5113579128470953, 'I2_slope': 0.3582543940065371, 'I2_RMS': 1.3019702907571222, 'Ihalf_slope': 0.47751379584239045, 'Ihalf_RMS': 1.0711061743738823, 'factual_weight': 10.0, 'insulin_weight': 1.0}, {'fact_RMS': 2.150537426282029, 'NE2_slope': 0.9705291353900561, 'NE2_RMS': 1.3453108520851642, 'NEhalf_slope': 2.578619427242916, 'NEhalf_RMS': 1.5178795048232694, 'I2_slope': 0.3601107161804689, 'I2_RMS': 1.3092235625721922, 'Ihalf_slope': 0.4776193405347022, 'Ihalf_RMS': 1.07794311849998, 'factual_weight': 100.0, 'insulin_weight': 10.0}, {'fact_RMS': 2.1544969603388395, 'NE2_slope': 0.9705051607737886, 'NE2_RMS': 1.3492805216867783, 'NEhalf_slope': 2.5835477482822227, 'NEhalf_RMS': 1.521545234273715, 'I2_slope': 0.3611609269605748, 'I2_RMS': 1.3179065908493919, 'Ihalf_slope': 0.4719596427168563, 'Ihalf_RMS': 1.08820959489592, 'factual_weight': 30.0, 'insulin_weight': 1.0}]

## Run 198: audit overbroad admission guard
Run197 used all pre-ICU bundles as exact-null constraints, consuming 180 directions. Audited event types and refitted with only initial pure admission bundles exactly protected; later factual bundles remain penalized. No weights changed. Python runs: 198; forwards: 22.
[{'fact_RMS': 0.34829411574766594, 'NE2_slope': 0.9173670231273487, 'NE2_RMS': 0.7942391088808117, 'NEhalf_slope': 1.4048536289291862, 'NEhalf_RMS': 0.7797662957351102, 'I2_slope': 0.11380674003765975, 'I2_RMS': 0.586885320091798, 'Ihalf_slope': 0.22625859715391033, 'Ihalf_RMS': 0.48985175377057366, 'factual_weight': 30.0, 'insulin_weight': 1.0}, {'fact_RMS': 0.36362974772777734, 'NE2_slope': 0.9240179152141543, 'NE2_RMS': 0.8967019446107913, 'NEhalf_slope': 1.6813112340213592, 'NEhalf_RMS': 0.9504311440734778, 'I2_slope': 0.1880579209804754, 'I2_RMS': 0.758147264076797, 'Ihalf_slope': 0.31319274778115624, 'Ihalf_RMS': 0.6211907003424508, 'factual_weight': 100.0, 'insulin_weight': 1.0}, {'fact_RMS': 0.3712978824466663, 'NE2_slope': 0.9139223883781422, 'NE2_RMS': 0.7255127532705383, 'NEhalf_slope': 1.1598842569758052, 'NEhalf_RMS': 0.6198339192268081, 'I2_slope': 0.03520684291447128, 'I2_RMS': 0.42694154788818495, 'Ihalf_slope': 0.13237859909077082, 'Ihalf_RMS': 0.3632551757218403, 'factual_weight': 10.0, 'insulin_weight': 1.0}, {'fact_RMS': 0.44892955418236385, 'NE2_slope': 0.910998624520756, 'NE2_RMS': 0.7027045000343739, 'NEhalf_slope': 1.028375280888642, 'NEhalf_RMS': 0.5095118593353032, 'I2_slope': -0.02704220167327054, 'I2_RMS': 0.31579202818940694, 'Ihalf_slope': 0.058755187404793775, 'Ihalf_RMS': 0.2620420252296437, 'factual_weight': 3.0, 'insulin_weight': 1.0}, {'fact_RMS': 0.5384771850859755, 'NE2_slope': 0.9267464785847348, 'NE2_RMS': 1.0587250965711774, 'NEhalf_slope': 2.344100416342337, 'NEhalf_RMS': 1.2751250216884773, 'I2_slope': 0.16580975980644436, 'I2_RMS': 0.609442734142738, 'Ihalf_slope': 0.23488200988897567, 'Ihalf_RMS': 0.5019246191813549, 'factual_weight': 100.0, 'insulin_weight': 10.0}]

## Run 199: install training-contrast NE trial
Corrected initial-admission guard leaves 188 input directions. Selected factual weight10 / insulin guard1: held-out factual RMS=.3713, NE x2 slope=.9139, half slope=1.1599, insulin slopes=.0352/.1324. Installed baseline77 plus .8*direction82 outer normalized feature. Saved ne_candidate199.npz; unaccepted pending GPU factual and rollout checks. Python runs: 199; forwards: 22.

## Run 200: reject candidate199 factual fit
Forward23: original NLL=1.63461, candidate199=1.65731; cost=.02270 exceeds .005. Candidate199 rejected. Recovering architecture, baseline schema, and prior causal/validation evidence before restoring validated weights. Python runs: 200; forwards: 23.

## Run 201: restore insulin baseline
Restored original core.b_in.weight plus baseline77 delta. Candidate199 remains archived but rejected for factual cost=.02270. Recovering prior validation trade-offs and persistent fitting objects. Python runs: 201; forwards: 23.

## Run 202: audit measured frontier and prior calibration
Insulin baseline remains installed. Aggregating all actual GPU factual costs and stay-averaged reached-hour dose effects; recovering calibration findings and exact reconstruction helpers to avoid repeating rejected approaches. Python runs: 202; forwards: 23.

## Run 203: investigate factual-detector failure mechanism
Baseline factual cost ranges .002783 to .004636 on two samples. Stay-averaged rollout differs from reached-hour-pooled averages; targets must be compared using identical weighting. Inspecting detector slot coverage and paired-state decoder contracts before selecting a new repair. Python runs: 203; forwards: 23.

## Run 204: rule out missing slot guards
Detector factual guards do cover all four slots. Recovering edit scales and prior decoder-calibration findings before changing direction or head. No weights changed; baseline77 installed. Python runs: 204; forwards: 23.

## Runs 205–206: correct calibration buffer mismatch
Run205 stopped before decoding: static builder returned 4751 validation rows rather than training rows, and its third output is register features. No weights changed. Run206 audits existing row-aligned training probabilities and tests scalar marginal calibration by held-out stays. Python runs: 206; forwards: 23.
{'pt_tune175': {'temperatures': [0.85, 0.85, 0.85, 0.85, 0.85], 'CV_NLL_change': -0.7137154021978116}}

## Run 207: reject misaligned calibration diagnostic
Run206 used tm162, which refers to tuning rows; its NLL near6 and apparent temperature gain are invalid and excluded. Run205 actually stopped on the third builder output being register features, not on row count. Explicitly aligning train_meta162 with paired train probabilities before any calibration decision. No weights changed. Python runs: 207; forwards: 23.
{'pt_edit175': {'temperatures': [0.9700000000000001, 0.9700000000000001, 0.9900000000000001, 0.9800000000000001, 0.9800000000000001], 'CV_NLL_change': -1.1760330528751971e-05}, 'pt_orig175': {'temperatures': [0.9700000000000001, 0.9700000000000001, 0.9900000000000001, 0.9800000000000001, 0.9800000000000001], 'CV_NLL_change': 3.801079986351693e-05}}

## Run 208: restarted-context recovery
Aligned temperature calibration offers no meaningful factual headroom. Recovering architecture, causal evidence, validated frontier, and persistent helpers before proposing another edit. Baseline77 remains installed. Python runs: 208; forwards: 23.

## Run 209: compact context recovery
Run208 output truncated before later evidence. Recovering targeted causal/validation findings, baseline archive schema, and recent fitting helpers with bounded output. No weights changed. Python runs: 209; forwards: 23.

## Run 210: bounded target and measured-frontier recovery
Insulin baseline passes narrowly and repairs the main response; NE evidence supports a positive effect but prior stronger edits fail factual fit. Recovering exact targets, construction archives, and GPU output schemas. No weights changed. Python runs: 210; forwards: 23.

## Run 211: strictly bounded artifact recovery
Working record-based targets: insulin about -13.6/+6.8; NE about +4.2..4.6/-2.1..2.3 on early tuning reached hours. Recovering saved edit schemas and measured forward outputs with a hard output bound. No weights changed. Python runs: 211; forwards: 23.

## Runs 212–213: measured GPU frontier and helper recovery
Run212 stopped on pandas shift-method collision; fixed with explicit column indexing. Aggregating factual costs and stay-weighted dose readouts; recovering decoder/detector contracts. No weights changed. Python runs: 213; forwards: 23.

## Run 214: align causal target weighting with evaluation
Baseline second-sample stay-weighted shifts: insulin -10.831/+3.357; NE +.098/-.955. Computing impulse-model targets on identical reached rows, averaging within stays then equally across stays. Parameter sensitivity is not ground truth. No weights changed. Python runs: 214; forwards: 23.

## Run 215: recover architecture and saved edit contracts
Matched stay-weighted targets make baseline insulin x2 reasonably close (-10.831 versus estimated -11.916 on sample2), but half remains weak (3.357 versus 5.958), and NE remains weak (+.098/-.955 versus +4.431/-2.216). Targets are record-derived impulse-model estimates, not observed counterfactual truth. Inspecting architecture and archives before choosing a lower-factual-cost edit. No weights changed; baseline77 remains installed. Python runs: 215; forwards: 23.

## Run 216: recover evidence and headroom options
Architecture confirms b_in is the recurrent state-update injection point and factual value predictions use a separate 17-class MLP plus register transition logits. Recovering baseline/NE validation and causal identification evidence, with bounded output. No weights changed. Python runs: 216; forwards: 23.

## Run 217: investigate insulin half-dose asymmetry
Prior NE shrinkage .4 still costs .00652 factual NLL, leaving little margin. Inspecting whether the accepted insulin detector was constrained mainly on x2 and can be refitted jointly on x2/half while retaining factual guards. No weights changed. Python runs: 217; forwards: 23.

## Run 218: assess partial insulin contrast balancing
Full balancing was previously attempted and cannot be reused blindly. Recovering candidate scales, measured factual losses, and fresh-training contrast buffers to assess a small correction rather than the rejected full edit. No weights changed. Python runs: 218; forwards: 23.

## Run 219: partial insulin balancing trial
Installed baseline77 + .15*(candidate63 delta*1.6/1.9 - baseline77), matching original baseline strength rather than using rejected candidate63 strength1.9. Intended x2 mean detector sensitivity is unchanged; half-dose sensitivity increases. Exact initial admission guard checked. Unaccepted pending GPU factual and rollout tests. Archived insulin_trial219.npz. Python runs: 219; forwards: 23.

## Run 220: recover current trial validation
Forward25 reports pooled reached-hour insulin shifts -12.474/+4.530 and NE -.203/-.794. These are predictions, not truth. Recovering Forward24 factual loss, architecture, and saved edit contracts before deciding whether trial219 is admissible. Python runs: 220; forwards: 25.

## Run 221: trial219 passes narrowly
Forward24 paired factual NLL: original 1.63461499, edited 1.63929569, cost .00468070. Trial219 is admissible on this sample but has only .00031930 margin. Recovering measured frontier, causal targets, and calibration history before further edits. Python runs: 221; forwards: 25.

## Run 222: target matching and fresh-validation preparation
Reviewing previous shrinkage/calibration outcomes and applying the impulse-model target to Forward25 reached rows. Selected previously unused tuning stays for a factual generalization check of trial219. Weights unchanged. Python runs: 222; forwards: 25.

## Run 223: fresh factual sample rejects narrow trial as robust choice
Forward26 NLL 1.63610 versus original 1.63043, cost .00567 on 100 unused tuning stays. Trial219 therefore lacks robust factual margin. Investigating closed-form leakage reduction or output-head distillation with causal-direction guards; no new weights installed. Python runs: 223; forwards: 26.

## Run 224: inspect guarded closed-form head correction feasibility
Current insulin edit is rank one. Considering a least-squares output-layer residual correction from paired original/baseline factual states, with directional guards to preserve causal responses. Inspecting decoder and row alignment before any fit. No weights changed. Python runs: 224; forwards: 26.

## Run 225: capture aligned paired value-head hidden features
Captured original and insulin-baseline hidden activations for 4510 training readings with four shared prior draws. Preparing a closed-form residual correction and stay-held-out tests, with no gradients and no weight edits. Python runs: 225; forwards: 26.

## Run 226: recover paired-head correction contracts
Run225 accurately reconstructed paired decoder probabilities. Inspecting aligned activation buffers and exact value-head helpers before fitting a closed-form correction. No weights changed. Python runs: 226; forwards: 26.

## Run 227: guarded factual-head distillation, offline only
Fitted ridge corrections to paired original/baseline factual logits, with entire stays held out and optional protection of dominant edit-induced hidden directions. No weights changed.
 rank  ridge  fit_vs_original  hold_vs_original  hold_gain  relative_weight_norm  paired_direction_change
    0    1.0         0.003267          0.001419  -0.003559              0.429418                 0.078499
    0   10.0         0.003325          0.001627  -0.003350              0.303101                 0.073515
    0  100.0         0.003503          0.002203  -0.002774              0.178118                 0.053921
    0 1000.0         0.003517          0.003017  -0.001961              0.070984                 0.022270
    4    1.0         0.003178          0.001179  -0.003798              0.379428                 0.030842
    4   10.0         0.003212          0.001377  -0.003600              0.246277                 0.029592
    4  100.0         0.003356          0.001966  -0.003011              0.149423                 0.023292
    4 1000.0         0.003439          0.002869  -0.002108              0.065197                 0.012504
   16    1.0         0.003353          0.001185  -0.003792              0.398040                 0.019766
   16   10.0         0.003380          0.001366  -0.003612              0.251038                 0.018004
   16  100.0         0.003476          0.002035  -0.002942              0.151371                 0.012252
   16 1000.0         0.003422          0.002963  -0.002015              0.064209                 0.006084
   32    1.0         0.003216          0.001384  -0.003594              0.411458                 0.012632
   32   10.0         0.003249          0.001574  -0.003404              0.231592                 0.011286
   32  100.0         0.003298          0.002129  -0.002849              0.144588                 0.007986
   32 1000.0         0.003114          0.002927  -0.002050              0.064653                 0.004181
Python runs: 227; forwards: 26.

## Runs 228–229: fresh-latent validation of head corrections
Run228 stopped on a diagnostic NpzFile attribute before decoding. Run229 uses paired fwd22 training states and 32 fresh shared prior draws. Corrections remain fitted only on run227 fit stays. No weights changed.
 rank  ridge  fit_vs_original  hold_vs_original  all_vs_original  all_gain  paired_direction_change
    0    1.0         0.003433          0.001667         0.003079  0.000300                 0.079100
    0   10.0         0.003477          0.001899         0.003161  0.000381                 0.074077
    0  100.0         0.003606          0.002543         0.003393  0.000613                 0.054329
    0 1000.0         0.003559          0.003291         0.003506  0.000726                 0.022404
    4    1.0         0.003268          0.001352         0.002884  0.000105                 0.030739
    4   10.0         0.003301          0.001563         0.002952  0.000173                 0.029463
    4  100.0         0.003433          0.002200         0.003186  0.000406                 0.023177
    4 1000.0         0.003483          0.003086         0.003404  0.000624                 0.012458
   16    1.0         0.003423          0.001267         0.002991  0.000211                 0.019884
   16   10.0         0.003448          0.001454         0.003048  0.000268                 0.018085
   16  100.0         0.003518          0.002150         0.003244  0.000464                 0.012270
   16 1000.0         0.003414          0.003065         0.003344  0.000564                 0.006070
   32    1.0         0.003184          0.001569         0.002860  0.000081                 0.012591
   32   10.0         0.003220          0.001750         0.002925  0.000146                 0.011245
   32  100.0         0.003281          0.002329         0.003090  0.000310                 0.007982
   32 1000.0         0.003096          0.003121         0.003101  0.000321                 0.004196
Python runs: 229; forwards: 26.

## Run 230: probability-weighted guarded head regression
Closed-form weighted residual regression fitted on eight latent draws and evaluated on 24 separate draws, with stay holdout. No weights changed.
 rank  strength  fit_vs_original  hold_vs_original  all_vs_original  fit_gain  hold_gain  all_gain  relative_weight_norm  paired_direction_change
   16      0.01         0.002688          0.004115         0.002974  0.000491  -0.001272  0.000137              0.068190                 0.006320
   16      0.10         0.002451          0.004425         0.002847  0.000254  -0.000962  0.000010              0.021456                 0.002095
   16      1.00         0.002267          0.004985         0.002812  0.000070  -0.000402 -0.000025              0.005762                 0.000604
   16     10.00         0.002169          0.005310         0.002799 -0.000028  -0.000077 -0.000038              0.001678                 0.000202
   32      0.01         0.002600          0.004003         0.002881  0.000403  -0.001384  0.000045              0.073941                 0.004461
   32      0.10         0.002389          0.004360         0.002784  0.000191  -0.001027 -0.000053              0.023651                 0.001480
   32      1.00         0.002255          0.005008         0.002807  0.000058  -0.000380 -0.000030              0.006704                 0.000428
   32     10.00         0.002155          0.005339         0.002793 -0.000043  -0.000048 -0.000044              0.001890                 0.000132
Python runs: 230; forwards: 26.

## Run 231: reject negligible head gain; restore baseline77
Probability-weighted head regression offered no material overall factual gain, so no head edit is accepted. Rejected trial219 after fresh-sample factual cost .00567 and restored baseline77 exactly. Auditing measured NE frontier for a conservative add-on.
Python runs: 231; forwards: 26.

## Run 232: prepare exact fresh-sample baseline comparison
Recovered the 100 stays from Forward26. Trial219 and baseline77 differ by only .000045 NLL on the first 100 tuning stays, so the fresh-sample excess cannot yet be attributed to balancing. Baseline77 remains installed; exact paired baseline validation next.
Python runs: 232; forwards: 26.

## Run 233: baseline fails fresh factual validation
Forward27 baseline77 NLL=1.63635 versus original=1.63043, cost .00592 on the same fresh 100 tuning stays; baseline itself lacks factual safety. Recovering architecture, record-derived targets, and prior calibration frontier before choosing a smaller or more selective edit. No weights changed. Python runs: 233; forwards: 27.

## Run 234: bounded calibration recovery
Recovered early identification: insulin RD/impulse estimates were negative despite positive raw association; NE history-adjusted and residual-dose estimates were positive despite negative raw association. Retrieving later targets and tested candidates with bounded output; no weights changed. Python runs: 234; forwards: 27.

## Run 235: exact measured frontier audit
Working early-tuning reached-hour targets: insulin about -13.6/+6.8; NE about +4.2..4.6/-2.1..2.3. Parsed archived paired factual costs and both stay-averaged and pooled dose shifts, and inspected insulin balancing candidate contracts. No weights changed. Python runs: 235; forwards: 27.

## Run 236: conservative joint repair trial
Installed original b_in + .85*((1-.35)*baseline77 + .35*candidate63_delta*1.6/1.9) + .20*ne_candidate151_ne_delta. Trial219 reconstruction max error=0. Saved exact composition in joint236.npz. Unaccepted pending paired factual validation on the fresh Forward27 stays and intervention validation. No gradients or weight optimizer. Python runs: 236; forwards: 27.

## Run 237: joint trial validated; inspect more selective route
Forward28 joint236 factual cost .00441 on the challenging fresh sample. Forward29 pooled shifts insulin -11.847/+4.000, NE +.115/-.882: causal repair remains incomplete, especially NE doubling. Joint236 retained. Inspecting prelude FFN feature access for a wider, nonlinear, closed-form dose-contrast detector with factual guards. No additional weights changed. Python runs: 237; forwards: 29.

## Run 238: nonlinear prelude route inspection
Joint236 retained unchanged. Latest paired factual cost .00441; pooled shifts insulin -11.847/+4.000, NE +.115/-.882. Inspecting architecture, nonlinear prelude helpers, retained feature arrays and saved NE detector contracts before any new edit. Hypothesis: a richer detector could increase intervention sensitivity without increasing factual disturbance; this remains untested.
Python runs: 238; forwards: 29.

## Run 239: compact recovery after truncated output
Architecture confirms a 768-wide GELU prelude FFN. Recovering helper return contracts, retained arrays and prior NE method notes with bounded output; joint236 remains unchanged.
Python runs: 239; forwards: 29.

## Run 240: audit prior nonlinear attempt before repeating it
Run93 already fitted 768-wide factual-centered prelude detectors. Recovering its validation outcome and later training-state feature contracts. Joint236 unchanged. Any new fit must demonstrate held-out dose sensitivity and factual safety rather than merely added capacity.
Python runs: 240; forwards: 29.

## Run 241: identify a genuinely different nonlinear fit
Corrected run94 features exist, but its fit gave highly asymmetric NE x2/half sensitivity. The old nonlinear exact-admission guard removed 116 of 769 directions. Investigating a training-state fit with only pure initial admissions guarded, balanced x2/half dose targets and explicit insulin guards. No weights changed.
Python runs: 241; forwards: 29.

## Run 242: training nonlinear features and alignment audit
Extracted prelude GELU features on baseline77 training states. Checking normalized reconstruction, posterior errors and synthetic dose-contrast encoding contracts before closed-form fitting. Joint236 unchanged.
Python runs: 242; forwards: 29.

## Run 243: GPU-anchored dose-feature reconstruction
Run242 normalized prelude reconstruction was exact, but absolute posterior reconstruction errors were material (baseline all RMS .3167; drug RMS .04390). Testing exact factual atom/set conventions and anchoring synthetic posterior changes at GPU factual z. No weights changed; joint236 retained.
Python runs: 243; forwards: 29.

## Run 244: balanced nonlinear NE detector cross-validation
Factual atom conventions and GPU-anchored dose contrasts verified exactly. Fitted 768-feature affine detectors by closed-form ridge, protecting only pure initial admissions and guarding both insulin contrasts. Five-fold whole-training-stay diagnostics:
 factual_weight  ridge  factual_RMS  NE2_slope  NE2_RMS  NEhalf_slope  NEhalf_RMS  I2_slope   I2_RMS  Ihalf_slope  Ihalf_RMS  unit_factual  half_ratio
            1.0  0.001     0.109005   0.438486 0.326305      0.325993    0.152956  0.000984 0.022636     0.005261   0.022622      0.248593    0.743451
            1.0  0.010     0.113922   0.373718 0.287929      0.294702    0.148222 -0.000211 0.016949     0.006881   0.020423      0.304833    0.788567
            1.0  0.100     0.113006   0.259176 0.207788      0.263393    0.138499  0.000219 0.019763     0.006759   0.027094      0.436018    1.016270
           10.0  0.001     0.064446   0.375805 0.281021      0.278127    0.132640  0.001621 0.021031     0.004753   0.021103      0.171487    0.740085
           10.0  0.010     0.062122   0.289034 0.224015      0.249293    0.124802  0.001879 0.017570     0.006661   0.019865      0.214931    0.862504
           10.0  0.100     0.056647   0.178787 0.146563      0.212587    0.110814  0.004206 0.018595     0.007471   0.022509      0.316841    1.189054
          100.0  0.001     0.032080   0.236063 0.178302      0.140156    0.067762  0.002291 0.017190     0.003362   0.017040      0.135896    0.593723
          100.0  0.010     0.025356   0.157905 0.121865      0.120049    0.059716  0.002939 0.015427     0.004677   0.015689      0.160580    0.760262
          100.0  0.100     0.019416   0.085026 0.069623      0.098921    0.050460  0.004542 0.014377     0.005882   0.014323      0.228353    1.163417
No weights changed. Python runs: 244; forwards: 29.

## Run 245: recover nonlinear edit contracts
Run244 whole-stay validation shows positive NE doubling/halving sensitivity with small insulin leakage; nonlinear weight edit is not yet tested. Inspecting retained coefficients and architecture before a conservative GPU trial. Joint236 remains installed. Python runs: 245; forwards: 29.

## Run 246: detector-to-weight mapping audit
Confirmed the nonlinear route uses prelude FFN output weights (192x768) and bias. Recovering detector scaling, output directions and prior nonlinear trial outcomes before editing. Joint236 unchanged. Python runs: 246; forwards: 29.

## Run 247: verify full nonlinear detector and local update mapping
Checking full-fit scaling, admission nulls, and NE/insulin contrast signs; recovering the forward-only prelude/coda helpers before choosing an output direction. No weights changed. Python runs: 247; forwards: 29.

## Run 248: correct coefficient-convention audit
Run247 incorrectly treated fits244 as standardized coefficients. Testing the raw-hidden-feature convention before any edit, and recovering the local prelude-to-state computation. Joint236 unchanged. Python runs: 248; forwards: 29.

## Runs 249-250: response-context recovery
Run249 stopped on a closed npz during inspection; no edits changed. Run250 skips closed archives and saves the correctly scaled nonlinear detector. Raw-coefficient factual RMS=.02323, exact admission nulls, positive NE2/half sensitivity. Recovering forward-only response contexts. Python runs: 250; forwards: 29.

## Run 251: anchored forward-difference route
Full recurrent absolute reconstruction is not trusted (prior best raw h RMSE 5.77); the coda inversion has exact GPU replay. Recovering this anchor and small-context reading interface for a MAP-specific prelude response direction. Joint236 unchanged. Python runs: 251; forwards: 29.

## Run 252: recover exact nonlinear-edit interfaces
Joint236 retained. GPU-anchored coda replay is accurate; full recurrent absolute reconstruction is not trusted. Inspecting architecture, saved nonlinear detector, and callable contracts before a forward-difference MAP-specific direction fit. No weights changed.
Python runs: 252; forwards: 29.

## Run 254: anchored recurrent differential interfaces
Recovered measured b_in MAP directions; nonlinear edit remains untested. Inspecting attention/memory and selected-state alignment for a differential recurrent calculation anchored at exact GPU pre-coda states. Joint236 unchanged.
Python runs: 254; forwards: 29.

## Runs 255-258: anchored prelude direction
Early attempts stopped on vocabulary conventions before fitting. Run258 verifies both query labels directly (8 MAP, 23 glucose), computes central sensitivities and closed-form guarded MAP directions, and checks perturbation size and memory convention. Saved prelude_direction255.npz. Joint236 unchanged.
Python runs: 258; forwards: 29.

## Run 259: restart and trial preparation
Local prelude MAP direction is stable but has glucose leakage; no GPU validation yet. Recovering architecture, causal evidence and saved edit contracts before deciding on a trial. Joint236 remains unchanged. Python runs: 259; forwards: 29.

## Run 260: bounded recovery
Architecture and initial record-based identification recovered. Printing compact target evidence and exact saved coefficients after previous output truncation. No weights changed. Python runs: 260; forwards: 29.

## Run 261: calibrate nonlinear trial units
Saved nonlinear detector and normalized MAP direction recovered. Checking actual dose-contrast units and prelude output tensor names before a small trial. No weights changed. Python runs: 261; forwards: 29.

## Run 262: first conservative nonlinear trial
Verified raw coefficient mapping. Added rank-one prelude FFN output weight/bias edit at scale 2.03173, targeting +0.20 mmHg local NE-doubling response. Joint236 b_in retained. Trial is unaccepted pending paired factual and rollout tests. Python runs: 262; forwards: 29.

## Run 263: reject full nonlinear trial; test reduced strength
Forward30 factual increase=.02294 on tuning100..199, unacceptable. Local anchored estimates missed full-model disturbance. Reduced prelude trial to 15% strength, retaining joint236 b_in. This version is unaccepted pending paired validation. Python runs: 263; forwards: 30.

## Run 264: restart audit
Forward32 reduced nonlinear rollout: insulin2=-11.289, insulinhalf=5.474, NE2=.103, NEhalf=-1.067 on tuning100. Recovering architecture, record-derived targets, forward31 factual fit, and accepted candidate contracts before retaining or rejecting the prelude edit. No weights changed.
Python runs: 264; forwards: 32.

## Run 265: compact validation and target recovery
Run264 output truncated before factual31 and saved candidate contracts. Recovering these compactly; no weights changed.
Python runs: 265; forwards: 32.

## Run 266: reject reduced nonlinear edit
Forward31 factual cost=.00571319, exceeding the .005 limit. Dropped both prelude edits and restored exact joint236 b_in. Reviewing earlier NE trials and factual/rollout frontier before another trial.
Python runs: 266; forwards: 32.

## Run 267: measured-frontier audit
Joint236 restored. Strong NE151 had NE2=1.893 and half=-1.633 but factual cost=.01107 with full insulin baseline. Recovering prior compensation outcomes and insulin balancing contracts before testing a conservative strength reallocation. No weights changed.
Python runs: 267; forwards: 32.

## Run 268: safer balanced joint trial
Forward33 joint236 factual cost=0.00523289 on tuning100..199; this exceeds .005, so joint236 is not robustly safe. Installed I_scale=.76, I_blend=.60, NE_scale=.40 using existing record-derived closed-form corrections; no prelude edits. Saved joint268.npz. Unaccepted pending paired factual and rollout tests.
Python runs: 268; forwards: 33.

## Run 269: restart audit of joint268
Forward35 tuning100 rollout: insulin2=-10.852, insulinhalf=5.236, NE2=.593, NEhalf=-.914. Reading architecture, record-derived identification, paired factual34 and saved joint contracts before accepting any edit. No weights changed. Python runs: 269; forwards: 35.

## Run 270: compact factual and target audit
Run269 was truncated before factual34. Retrieving current paired factual NLL, cluster-bootstrap uncertainty, saved joint contracts and record-derived targets. No weights changed. Python runs: 270; forwards: 35.

## Run 271: joint268 rejected on factual fit
Forward34 original NLL=1.620435206, edited=1.625789773, delta=.005354567 (fails .005). By-event degradation: glucose_finger=.005925, glucose_lab=.018507, map_arterial=.003620, map_cuff=.001913. Bootstrap interval [.002196,.008817]. Recovering safer candidate and prior calibration contracts. joint268 remains installed only for diagnostics, not accepted. Python runs: 271; forwards: 35.

## Run 272: recover final causal targets and cohort alignment
Current joint268 factual mean drift is not a simple calibration error; prior aligned temperature calibration gave no meaningful headroom. Reviewing final identification and measured validation cohorts before conservative recurrent-strength reduction. No weights changed. Python runs: 272; forwards: 35.

## Run 273: cohort targets and measured NE frontier
Recovered corrected record-only NE dynamics: beta≈3.17, phi≈.704; early NE2 reached-hour target≈4.2–4.6, half≈-2.1–-2.3. Inspecting exact current-cohort targets and previously measured candidates before final safety-oriented trials. No weights changed. Python runs: 273; forwards: 35.

## Run 274: conservative joint shrinkage trial
Forward35 exact-cohort impulse targets: insulin2=-14.149, half=7.074; NE2=4.607, half=-2.304 (statistical estimates, not hidden truth). Previously measured NE alternatives were less factual-efficient. Installed 0.90*joint268 delta: I_scale=.684, I_blend=.60, NE_scale=.36. Saved joint274.npz. Only core.b_in.weight edited; no gradients or optimizer. Unaccepted pending paired factual validation across tuning cohorts. Python runs: 274; forwards: 35.

## Run 275: cross-cohort audit of joint274
Forward37 reported factual delta about .00206 on 100 tuning stays, below .005. Recovering exact forward36/37 evidence, architecture and saved candidate contracts before acceptance or further edits. Weights unchanged. Python runs: 275; forwards: 37.

## Run 276: compact audit after truncated run275
Architecture confirms the current edit acts solely through core.b_in.weight. Joint236/268 exceeded .005 on tuning100..199. Recovering compact paired results, rollout schema and record-identification contracts. No edits changed. Python runs: 276; forwards: 37.

## Run 277: paired evidence and candidate frontier
Joint274 factual deltas: cohort100..199=.0043248285, cohort300..399=.0020602652; pooled about .00319. Recovering equal-stay rollout means, candidate decompositions and past compensation evidence before spending remaining validation passes. No weights changed. Python runs: 277; forwards: 37.

## Run 278: full-cohort safety and insulin-prioritized trial
Joint274 all400 factual delta=0.002951029; cluster bootstrap95=[0.0014079411070059816, 0.004518309643760527]. All four 100-stay cohorts were individually below .005. Installed unaccepted joint278: I_scale=.90, I_blend=.60, NE_scale=.25; only core.b_in.weight. Closed-form recombination of existing record-derived rank corrections, no gradients/optimizer. Joint274 remains archived as the validated factual fallback. Python runs: 278; forwards: 39.

## Run 279: reject stronger insulin trial on measured cohort
Forward40 joint278 factual delta approximately .00672 on tuning100..199, above .005. Not accepted. Recovering architecture, archived candidate contracts and exact loss before restoring fallback or constructing a smaller insulin-prioritized trial. No weights changed in this run.
Python runs: 279; forwards: 40.

## Run 280: restored validated fallback
Exact forward40 joint278 factual delta=.00672373499 on tuning100..199. Verified archive convention and restored joint274 (I=.684, blend=.60, NE=.36), whose all400 factual delta=.002951029. Reviewing measured factual/rollout frontier before further trials.
Python runs: 280; forwards: 40.

## Run 281: measured response/cost frontier
Joint274 remains installed. Recomputed past rollout summaries using only reached hours13..24, averaging within stay before across stays. Compared paired event-specific factual costs on the identical tuning100..199 cohort and recovered early record-identification evidence. No weights changed.
Python runs: 281; forwards: 40.

## Runs 282-283: corrected diagnostic and candidate A
Run282 failed on pandas GroupBy.shift name collision before editing. Run283 uses explicit column indexing, audits rollout aggregation, and installs unaccepted A: I=.78, blend=.35, NE=.20. Closed-form recombination of joint236 corrections; no gradients. Saved joint283.npz. Joint274 remains the validated fallback. Next paired factual and dose validation on tuning100..199.
Python runs: 283; forwards: 40.

## Run 284: candidate A measured; candidate B installed
Aggregation resolved: evaluator averages reached stay-hours, not equal-stay means. Forward41 A factual delta=0.0044366423670669786; forward42 reached-hour effects={"insulin_half": 5.153301026694045, "insulin_x2": -10.592675975359343, "ne_half": -0.8474376666666666, "ne_x2": 0.327277}. A improves all four original responses but still underestimates effects. Installed unaccepted B: I=.80, blend=.60, NE=.25, saved joint284.npz. Closed-form recombination of archived directions, no gradients/optimizer. Plan: paired factual across all400 tuning stays and exact rollout. Judge aggregate fit with uncertainty; one-cohort failure alone does not establish population failure. Joint274 remains validated fallback.
Python runs: 284; forwards: 42.

## Run 285: restart audit before population validation
Candidate B remains installed and unaccepted. Recovered architecture, record-identification notes, archive contracts, and tuning schema. No weights changed. Next: paired factual validation across all400 tuning stays and exact dose rollout; retain only if population fit supports the constraint.
Python runs: 285; forwards: 42.

## Run 286: compact validation-contract audit
Previous output truncated before archives and IDs. Retrieved candidate tensor contracts, checked installed/archive distances, and recovered tuning ID ranges and factual schemas. No weights changed.
Python runs: 286; forwards: 42.

## Run 287: verify installed candidate and pooled fallback
Checked original-plus-archived-delta reconstruction and component conventions. Recomputed joint274 paired factual fit across all400 tuning stays, including stay-cluster bootstrap. No weights changed. Candidate B remains unaccepted pending all400 factual validation and paired rollout.
Python runs: 287; forwards: 42.

## Run 288: candidate B all400 paired factual audit
B cohort costs={43: 0.005370261578044595, 44: 0.0037517575247316345, 45: 0.003330681569143112, 46: 0.0021559452523524382}. All400 NLL increase=0.0036534041; cluster-bootstrap95=[0.001962264794154504, 0.005381214034772753]; one-sided95 upper=0.0050963395. The pooled observed mean meets .005, but uncertainty must be considered. No weights changed; candidate B still needs exact dose-rollout validation. Only core.b_in.weight is edited, exactly original plus joint284 delta.
Python runs: 288; forwards: 46.

## Run 289: final-budget evidence audit
Forward47 candidate B reached-hour effects: insulin x2=-11.312, half=5.383; NE x2=.411, half=-.720. B all400 factual cost=.0036534041, one-sided95 upper=.0050963395. Reviewing architecture, record-only identification and archived safer alternatives before allocating three remaining forwards. No weights changed.
Python runs: 289; forwards: 47.

## Run 290: bounded archive and identification audit
Run289 output truncated before archive contracts. Retrieved compact candidate metadata, installed recurrent-weight delta, reusable kernel objects and selected identification/construction notes. No weights changed.
Python runs: 290; forwards: 47.

## Run 291: conservative final candidate C installed
Verified installed B=original+joint284 delta (max error=0.0). Installed 0.95*B delta: insulin_scale=.76, blend=.60, NE_scale=.2375. Only core.b_in.weight changed; saved joint291.npz. This is closed-form shrinkage of record-derived corrections, with no gradients or optimizer. C is unaccepted pending paired factual checks and exact dose rollout; joint274 remains the all400-validated fallback.
Python runs: 291; forwards: 47.

## Run 292 diagnostic failure
Stopped at cross-request original-NLL equality assertion before any weight change or audit conclusions. Within-request original/edited alignment checks had passed. Cross-request bit equality is not required by the tool contract; correcting that assumption in run293.
Python runs: 292; forwards: 49.

## Run 293: corrected final-C factual audit
Event identities align across requests. Forward48 C delta=0.0048429245; forward49=0.0033008209. Pooled observed delta=0.0040647477, 9415 readings / 200 stays. Both cohort means pass .005. Selected-200 bootstrap95=[0.0016326991268820433, 0.006546070668411839]; these were B's two highest-cost cohorts, not a random population sample. C/B cost ratio=0.892658; all400 scalar extrapolation=0.0032612415, explicitly not measured. Archive verified; SHA256=c93a8cf4fad68f4cec97c3f7c5cc1c2ceea479611ba573bd115eb6df4f99bbc2. No weights changed. Final forward reserved for C dose rollout on tuning100..199.
Python runs: 293; forwards: 49.

## Run 294: final-rollout preparation
Recovered architecture, identification notes, candidate archive contracts and the exact forward48 cohort. No weights changed. C remains provisional pending the final paired dose rollout. Python runs: 294; forwards: 49.

## Run 295: compact final preflight
Verified unchanged candidate C by SHA256 and sole edited tensor. Recovered exact forward48 rollout IDs and compact record-effect notes. No weights changed. Final paired dose rollout is next. Python runs: 295; forwards: 49.

## Run 296: final choice and submission audit
Forward50 exact paired rollout on tuning100..199: original insulin x2 -1.879, half +0.868; NE x2 -0.312, half -0.647. Candidate C: insulin x2 -11.017, half +5.130; NE x2 +0.363, half -0.741. Keep C unchanged: main insulin correction retained; NE doubling sign repaired, but NE magnitudes remain substantially below record-derived estimates. No further unvalidated edits.
Final factual evidence: {"n_stays": 200, "n_readings": 9415, "original_nll": 1.6275906425916091, "edited_nll": 1.6316553903345727, "increase": 0.004064747742963357}. This is measured on selected200 tuning stays, not unseen evaluation stays; all400 C factual fit was not measured. B all400 and C's shrinkage support, but do not guarantee, safety.
Only core.b_in.weight edited, original plus archived joint291 delta; closed-form record-derived correction, no gradients or optimizer. Final SHA256: c93a8cf4fad68f4cec97c3f7c5cc1c2ceea479611ba573bd115eb6df4f99bbc2. Python runs: 296; forwards: 50. Ready for final submission.
