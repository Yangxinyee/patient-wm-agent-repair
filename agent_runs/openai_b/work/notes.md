# Weight-repair notebook

## Objective and constraints
Infer simulated drug effects exclusively from training records. Edit weights without model gradients or an optimizer. Match insulin x2/x0.5 glucose and norepinephrine x2/x0.5 MAP rollouts; keep factual value NLL deterioration <= 0.005 nats/reading.

## Initial plan
Read architecture/schema; investigate treatment assignment, temporal ordering, and identifiable dose variation; estimate effects with small statistical models; locate an editable treatment-response mechanism; validate factual fit and rollouts on tuning stays.
No drug-effect direction or magnitude assumed.

## Run 1: architecture
Separate DeepSets action encoder; action influences posterior latent and prelude/state. Value head is linear on with_code plus a per-code/last-bin transition table. Numeric embedding jointly uses value, dose and rate. Same-code last-reading register is important. No causal conclusion yet. Initial output was truncated; run 2 loads records and prints focused diagnostics.

## Run 2: records
2,057,535 events, 30,000 training stays; 400 tuning stays. Readings at integer hours -1..24; actions at h+0.0833, end one hour later. Insulin in 18,508 stays, norepinephrine in 13,200. Sensors have substantial missingness and different populations. Run 3 constructs hourly panels and inspects policy/transition associations; these are explicitly not causal estimates yet.

## Run 3: preliminary associations
Naive next-reading slopes: insulin +24.48 per unit/hour, norepinephrine -35.58 per mcg/kg/min. Current-reading cubic adjustment gives +0.166 and -2.415 respectively; not causal. Insulin policy has large steps near glucose 150,180,220,260,300 with within-bucket rate variation. Investigating thresholds, sensor information and dose persistence before concluding.

## Runs 4 and baseline forwards
Insulin exact-current FE slope 0.034 (stay-cluster SE 0.0426); rich history-adjusted slope 0.0105, lag -0.0113. Norepinephrine exact-current slope -2.5509; rich history slope -0.3866, lag +0.117. Norepinephrine titration is persistent and some contemporaneous sensors are missing, so remaining confounding is plausible; no final conclusion yet.
Forward 1 (files fwd_2; one rejected duplicate-ID request before it): original tuning first100 rollouts insulin x2 +0.209, half -0.072; NE x2 -0.007, half +0.032. Forward 2 (fwd_3): original factual NLL 1.50467 on 4,822 readings.
Run 5 cross-fitted nonlinear pre-treatment-history estimates: {"insulin": {"beta": 0.017539510905785215, "cluster_se": 0.0362424457963426, "n": 327225, "residual_rate_sd": 0.7432381573771983}, "norepinephrine": {"beta": -0.1779133490910641, "cluster_se": 0.08104593589925974, "n": 312563, "residual_rate_sd": 0.09672071796371311}}

## Restart / run 6
Insulin cross-fit estimate is compatible with zero; norepinephrine's remaining small negative association needs independent identification checks. Re-inspecting architecture and retained analysis objects before designing a surgical weight edit. No weights edited yet.

## Run 8: independent sensors and delayed-response checks
[
  {
    "drug": "insulin",
    "outcome": "glucose_finger",
    "horizon": 1,
    "beta": 0.018741809745798515,
    "cluster_se": 0.03628329932122163,
    "n": 327225
  },
  {
    "drug": "insulin",
    "outcome": "glucose_finger",
    "horizon": 4,
    "beta": -0.0241329642710366,
    "cluster_se": 0.050381420715773194,
    "n": 275472
  },
  {
    "drug": "insulin",
    "outcome": "glucose_finger",
    "horizon": 8,
    "beta": -0.04030081051651384,
    "cluster_se": 0.07221265297558951,
    "n": 218857
  },
  {
    "drug": "insulin",
    "outcome": "glucose_lab",
    "horizon": 1,
    "beta": 0.17126519319891562,
    "cluster_se": 0.10447169796054676,
    "n": 28483
  },
  {
    "drug": "insulin",
    "outcome": "glucose_lab",
    "horizon": 4,
    "beta": 0.024272032759568605,
    "cluster_se": 0.17056120578213949,
    "n": 25154
  },
  {
    "drug": "insulin",
    "outcome": "glucose_lab",
    "horizon": 8,
    "beta": 0.3259282014882791,
    "cluster_se": 0.29124754206018333,
    "n": 20332
  },
  {
    "drug": "norepinephrine",
    "outcome": "map_arterial",
    "horizon": 1,
    "beta": -0.1369705377215296,
    "cluster_se": 0.08095979192869435,
    "n": 312563
  },
  {
    "drug": "norepinephrine",
    "outcome": "map_arterial",
    "horizon": 4,
    "beta": 0.154987712123549,
    "cluster_se": 0.11265807988136205,
    "n": 273503
  },
  {
    "drug": "norepinephrine",
    "outcome": "map_arterial",
    "horizon": 8,
    "beta": 0.499396068844614,
    "cluster_se": 0.16484943074177824,
    "n": 221503
  },
  {
    "drug": "norepinephrine",
    "outcome": "map_cuff",
    "horizon": 1,
    "beta": -0.4562282590430393,
    "cluster_se": 0.2063119316438813,
    "n": 333382
  },
  {
    "drug": "norepinephrine",
    "outcome": "map_cuff",
    "horizon": 4,
    "beta": -0.6025750084647972,
    "cluster_se": 0.2540175633977814,
    "n": 291606
  },
  {
    "drug": "norepinephrine",
    "outcome": "map_cuff",
    "horizon": 8,
    "beta": -0.4644222045968939,
    "cluster_se": 0.3250557715913275,
    "n": 236222
  }
]

## Run 9: candidate A, moment-preserving dose-channel ablation
Zeroed numeric dose/rate magnitude columns, transferred their empirical mean contributions to the corresponding presence columns, and made nonmissing dose/rate bins equal their empirical average embedding. Missing bins and measurement-value channels are unchanged. This is a reversible three-tensor edit intended to produce exact dose invariance; factual validation is pending.
Numeric means: {"dose": 0.05720048301019082, "rate": 0.05720048301019082}

## Forward passes 3–4: candidate A validation
First 100 tuning stays: original factual NLL 1.50467, edited 1.50465 (difference -0.00002). Dose-edit rollout confirms all four edited shifts exactly zero. Original shifts: insulin x2 +0.209, half -0.072; NE x2 -0.007, half +0.032.

## Run 10: causal robustness checks
{
  "strata": [
    {
      "drug": "insulin",
      "stratum": "hours_12_23",
      "beta": 0.041141039352419986,
      "cluster_se": 0.04863597509471053,
      "n": 174081,
      "dose_resid_sd": 0.7794382070066376
    },
    {
      "drug": "insulin",
      "stratum": "previously_running",
      "beta": 0.01433919100768343,
      "cluster_se": 0.03714703082665263,
      "n": 269511,
      "dose_resid_sd": 0.8008709786588177
    },
    {
      "drug": "insulin",
      "stratum": "both_current_sensors",
      "beta": 0.11541144572887575,
      "cluster_se": 0.07607975008145587,
      "n": 65369,
      "dose_resid_sd": 0.7490977547319616
    },
    {
      "drug": "insulin",
      "stratum": "running_and_both_sensors",
      "beta": 0.15329377182232123,
      "cluster_se": 0.0778158685930158,
      "n": 53764,
      "dose_resid_sd": 0.807920446213782
    },
    {
      "drug": "norepinephrine",
      "stratum": "hours_12_23",
      "beta": 0.03650343845490132,
      "cluster_se": 0.10889337426412993,
      "n": 156331,
      "dose_resid_sd": 0.0984400110848171
    },
    {
      "drug": "norepinephrine",
      "stratum": "previously_running",
      "beta": -0.07280226944869216,
      "cluster_se": 0.08385963607545956,
      "n": 142827,
      "dose_resid_sd": 0.13679162141055531
    },
    {
      "drug": "norepinephrine",
      "stratum": "both_current_sensors",
      "beta": -0.1370212563332267,
      "cluster_se": 0.1467881103039392,
      "n": 93413,
      "dose_resid_sd": 0.09699123632184221
    },
    {
      "drug": "norepinephrine",
      "stratum": "running_and_both_sensors",
      "beta": -0.005084975550080735,
      "cluster_se": 0.15313269623792478,
      "n": 42414,
      "dose_resid_sd": 0.13761419369329295
    }
  ],
  "threshold_iv": [
    {
      "bandwidth": 6,
      "horizon": 1,
      "dose_jump": 0.6839148181480885,
      "beta": 0.17843630911398184,
      "cluster_se": 0.2593234934646669,
      "n": 100328
    },
    {
      "bandwidth": 6,
      "horizon": 4,
      "dose_jump": 0.7526404468764533,
      "beta": -0.07071546428392844,
      "cluster_se": 0.31996999310357827,
      "n": 83756
    },
    {
      "bandwidth": 10,
      "horizon": 1,
      "dose_jump": 0.8194803402258397,
      "beta": -0.027755178910626253,
      "cluster_se": 0.17022438875090665,
      "n": 166840
    },
    {
      "bandwidth": 10,
      "horizon": 4,
      "dose_jump": 0.868103216851689,
      "beta": 0.046279646884333614,
      "cluster_se": 0.21418681501644954,
      "n": 139602
    },
    {
      "bandwidth": 14,
      "horizon": 1,
      "dose_jump": 0.8857545990711545,
      "beta": 0.006162705647334451,
      "cluster_se": 0.13355210122733477,
      "n": 234557
    },
    {
      "bandwidth": 14,
      "horizon": 4,
      "dose_jump": 0.921284448270816,
      "beta": -0.21498927461347997,
      "cluster_se": 0.17222746715848483,
      "n": 195926
    }
  ]
}

## Restart / run 11
Forward 5 (fwd_6), second 100 tuning stays: original factual NLL 1.49403, edited 1.49391, difference -0.00012. Together with first batch, no observed factual penalty. Candidate A still enforces exact dose-magnitude invariance. Next: recover architecture and analysis objects, then check nonlinear/high-dose effects and validate remaining tuning stays before accepting zero effects.

## Run 12: architecture audit
Dose edits change dose/rate numeric scores and fine-bin embeddings. The three edited tensors remove those magnitude paths while preserving code, presence, missingness, and measurement inputs. Reading-head registers are per measurement code, so drug-rate registers do not directly affect reading attributes. Inspecting residual-analysis objects for nonlinear checks and identifying the two remaining tuning batches.

## Forward passes 6–7 and run 13
Third tuning batch: NLL delta -0.00011; fourth: -0.00002. All 400 tuning stays now have paired factual validation.
Pooled factual summary:
{"stays": 400, "readings": 18910, "original_nll": 1.503084563722898, "edited_nll": 1.503016271813855, "delta": -6.829190904283489e-05, "stay_cluster_se": 2.799329639474757e-05, "delta_95ci": [-0.00012315876997654013, -1.3425048109129657e-05]}
Late-period, cross-fitted outcome-change slopes by predicted policy dose and sensor completeness (observational checks, not proof of identification):
[{"n": 174081, "beta": 0.03374542889347911, "cluster_se": 0.04860212256021768, "residual_rate_sd": 0.7791372279843655, "drug": "insulin", "horizon": 1, "stratum": "all_with_current_reading"}, {"n": 58473, "beta": -0.1489137942182986, "cluster_se": 0.15606259939627862, "residual_rate_sd": 0.2988584690423087, "drug": "insulin", "horizon": 1, "stratum": "low_policy_dose"}, {"n": 38841, "beta": 0.0687342041014357, "cluster_se": 0.06745058761384147, "residual_rate_sd": 1.2948916388862632, "drug": "insulin", "horizon": 1, "stratum": "high_policy_dose"}, {"n": 30307, "beta": 0.08395740529597316, "cluster_se": 0.10205815565783856, "residual_rate_sd": 0.8402045176504274, "drug": "insulin", "horizon": 1, "stratum": "previously_running_both_sensors"}, {"n": 126346, "beta": 0.009322944609199513, "cluster_se": 0.06983024029440933, "residual_rate_sd": 0.7683858816662239, "drug": "insulin", "horizon": 4, "stratum": "all_with_current_reading"}, {"n": 44160, "beta": -0.09388767732422368, "cluster_se": 0.24395166131515625, "residual_rate_sd": 0.2840396287558518, "drug": "insulin", "horizon": 4, "stratum": "low_policy_dose"}, {"n": 27459, "beta": -0.0024940373650027075, "cluster_se": 0.09716345380559505, "residual_rate_sd": 1.2783602402579546, "drug": "insulin", "horizon": 4, "stratum": "high_policy_dose"}, {"n": 21938, "beta": -0.0061259397596409694, "cluster_se": 0.15439608270931351, "residual_rate_sd": 0.8267333741441614, "drug": "insulin", "horizon": 4, "stratum": "previously_running_both_sensors"}, {"n": 156331, "beta": 0.036521138454017936, "cluster_se": 0.10935472802341716, "residual_rate_sd": 0.09816249107204253, "drug": "norepinephrine", "horizon": 1, "stratum": "all_with_current_reading"}, {"n": 100796, "beta": 0.9614423874370043, "cluster_se": 0.4074487265644309, "residual_rate_sd": 0.03366469225676959, "drug": "norepinephrine", "horizon": 1, "stratum": "low_policy_dose"}, {"n": 18512, "beta": -0.382695830005712, "cluster_se": 0.22396766955397673, "residual_rate_sd": 0.13945747024728078, "drug": "norepinephrine", "horizon": 1, "stratum": "high_policy_dose"}, {"n": 22076, "beta": 0.08648323582496788, "cluster_se": 0.1982024942540982, "residual_rate_sd": 0.14287101020558668, "drug": "norepinephrine", "horizon": 1, "stratum": "previously_running_both_sensors"}, {"n": 117228, "beta": 0.12252907960740042, "cluster_se": 0.15507214081318302, "residual_rate_sd": 0.09819275258518197, "drug": "norepinephrine", "horizon": 4, "stratum": "all_with_current_reading"}, {"n": 75451, "beta": 0.24651002872628294, "cluster_se": 0.5942518257491226, "residual_rate_sd": 0.03267978373042208, "drug": "norepinephrine", "horizon": 4, "stratum": "low_policy_dose"}, {"n": 13926, "beta": -0.1424237170535361, "cluster_se": 0.31394874466798334, "residual_rate_sd": 0.13947186391861377, "drug": "norepinephrine", "horizon": 4, "stratum": "high_policy_dose"}, {"n": 16663, "beta": -0.31820286400190795, "cluster_se": 0.28150401186644364, "residual_rate_sd": 0.1429033974392306, "drug": "norepinephrine", "horizon": 4, "stratum": "previously_running_both_sensors"}]

## Run 14: nonlinear magnitude checks
Cross-fitted linear/quadratic dose-response moments, restricted to late-period continuing positive infusions with both current sensors observed. Outcome is future minus current reading; controls are pre-action history and the other current drug. These are single-action association checks, not estimates of the evaluation's sustained-dose rollout. No extrapolation of quadratic curves to doubled doses is used.
[{"drug": "insulin", "horizon": 1, "n": 29845, "rate_scale": 1.0, "linear_beta": 0.40632727337140384, "linear_se": 0.20207875323817148, "quadratic_beta": -0.03656187085659365, "quadratic_se": 0.02319584781864689, "joint_p": 0.1236770181157683, "basis_condition_number": 446.3262946924651, "q25_rate": 1.0, "q25_slope": 0.33320353165821653, "q25_slope_se": 0.16312438229069726, "q75_rate": 3.0, "q75_slope": 0.18695604823184192, "q75_slope_se": 0.10521494103594256}, {"drug": "insulin", "horizon": 4, "n": 21292, "rate_scale": 1.0, "linear_beta": 0.3447529092098515, "linear_se": 0.32788521852889135, "quadratic_beta": -0.036734206121328584, "quadratic_se": 0.03981851681294282, "joint_p": 0.5750696015155612, "basis_condition_number": 430.338895691652, "q25_rate": 1.5, "q25_slope": 0.23455029084586576, "q25_slope_se": 0.22799154779619, "q75_rate": 3.0, "q75_slope": 0.12434767248188001, "q75_slope_se": 0.1580848894251216}, {"drug": "norepinephrine", "horizon": 1, "n": 21886, "rate_scale": 0.1, "linear_beta": 0.09803822663293406, "linear_se": 0.08506198137899076, "quadratic_beta": -0.00638832825265425, "quadratic_se": 0.005846012528742591, "joint_p": 0.5107959185453195, "basis_condition_number": 3761.9015639243485, "q25_rate": 0.41, "q25_slope": 0.4565393496116921, "q25_slope_se": 0.39993322732912195, "q75_rate": 0.9499999999999998, "q75_slope": -0.23340010167496664, "q75_slope_se": 0.34485213078676963}, {"drug": "norepinephrine", "horizon": 4, "n": 16523, "rate_scale": 0.1, "linear_beta": 0.23480695734499538, "linear_se": 0.1170203031412149, "quadratic_beta": -0.018538610207425744, "quadratic_se": 0.008064370543817723, "joint_p": 0.046329931078360476, "basis_condition_number": 3531.2353911629993, "q25_rate": 0.39, "q25_slope": 0.9020579772707459, "q25_slope_se": 0.5801231300127451, "q75_rate": 0.94, "q75_slope": -1.1371891455460854, "q75_slope_se": 0.4742247405480556}]

## Run 15: investigate marginal nonlinear signal
Run 14 NE horizon-4 quadratic joint p=0.046 was isolated among four checks; insulin joint p=0.124/0.575 and NE horizon-1 p=0.511. Testing richer nuisance models, all-running versus both-sensor subsets, and current-minus-previous reading as a pre-infusion placebo. Significant placebo moments diagnose estimation bias; null placebos cannot prove unconfoundedness.
Parameters: {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 80, "l2_regularization": 1.0, "early_stopping": false, "random_state": 2026}
[{"n": 29448, "linear_beta": -0.012039841762953276, "linear_se": 0.057632832965988604, "quadratic_beta": -0.003054113890772935, "quadratic_se": 0.00843114701350436, "joint_p": 0.192243822446122, "drug": "insulin", "horizon": 0, "stratum": "both_sensors_running", "rate_scale": 1.0, "outcome_type": "pre_infusion_placebo"}, {"n": 149273, "linear_beta": -0.10247476695820687, "linear_se": 0.10038770214592187, "quadratic_beta": 0.01595189821051556, "quadratic_se": 0.011760933591023127, "joint_p": 0.371758516589388, "drug": "insulin", "horizon": 1, "stratum": "all_running", "rate_scale": 1.0, "outcome_type": "future_change"}, {"n": 29845, "linear_beta": 0.5431494447973164, "linear_se": 0.1975845494142172, "quadratic_beta": -0.051521432012112754, "quadratic_se": 0.02226614528712956, "joint_p": 0.022565472550754702, "drug": "insulin", "horizon": 1, "stratum": "both_sensors_running", "rate_scale": 1.0, "outcome_type": "future_change"}, {"n": 106417, "linear_beta": 0.13137292156214364, "linear_se": 0.14410987678115575, "quadratic_beta": -0.01474290471844492, "quadratic_se": 0.016843627504361837, "joint_p": 0.65238815843347, "drug": "insulin", "horizon": 4, "stratum": "all_running", "rate_scale": 1.0, "outcome_type": "future_change"}, {"n": 21292, "linear_beta": 0.2950092134422848, "linear_se": 0.3203250089603941, "quadratic_beta": -0.03708075425968757, "quadratic_se": 0.038649865639630625, "joint_p": 0.6241422982719464, "drug": "insulin", "horizon": 4, "stratum": "both_sensors_running", "rate_scale": 1.0, "outcome_type": "future_change"}, {"n": 21931, "linear_beta": -0.000954083688706939, "linear_se": 0.00980537353628543, "quadratic_beta": -6.475312808931596e-05, "quadratic_se": 0.0006652277877004804, "joint_p": 0.6909813268918504, "drug": "norepinephrine", "horizon": 0, "stratum": "both_sensors_running", "rate_scale": 0.1, "outcome_type": "pre_infusion_placebo"}, {"n": 73500, "linear_beta": 0.019046988849695506, "linear_se": 0.04923054337470549, "quadratic_beta": -0.0012244749778461193, "quadratic_se": 0.003386835078253181, "joint_p": 0.9255139317970357, "drug": "norepinephrine", "horizon": 1, "stratum": "all_running", "rate_scale": 0.1, "outcome_type": "future_change"}, {"n": 21886, "linear_beta": 0.09739228975392902, "linear_se": 0.08108648656404534, "quadratic_beta": -0.006580598411873411, "quadratic_se": 0.0055623113436065194, "joint_p": 0.48487794330359935, "drug": "norepinephrine", "horizon": 1, "stratum": "both_sensors_running", "rate_scale": 0.1, "outcome_type": "future_change"}, {"n": 55285, "linear_beta": 0.10783673369423825, "linear_se": 0.06805985664955215, "quadratic_beta": -0.00707293230579355, "quadratic_se": 0.004687876949595348, "joint_p": 0.28191364251824047, "drug": "norepinephrine", "horizon": 4, "stratum": "all_running", "rate_scale": 0.1, "outcome_type": "future_change"}, {"n": 16523, "linear_beta": 0.20520138154917772, "linear_se": 0.11114027623275251, "quadratic_beta": -0.017007746330824233, "quadratic_se": 0.007689014327583921, "joint_p": 0.043261886776612184, "drug": "norepinephrine", "horizon": 4, "stratum": "both_sensors_running", "rate_scale": 0.1, "outcome_type": "future_change"}]

## Restart / runs 16–18
Recovered architecture and candidate A. Three input tensors edited: nonmissing dose/rate bin rows made constant; numeric magnitude columns zeroed with their mean contributions transferred to presence columns. Candidate A had exact zero dose-edit shifts on the first100 and first100 factual NLL delta approximately -0.00002. Repeated stay IDs caused a rejected forward request, not empirical evidence. Now aggregating four disjoint factual tuning batches and preparing unique IDs for a second rollout check.

## Final candidate audit / run 19 and forward 8
Kept candidate A unchanged, hash 7971f330d6. Independent second100 rollout cohort also gave exact zero for insulin_x2, insulin_half, ne_x2 and ne_half. Original second100 shifts were +0.141, -0.218 mg/dL and -0.002, +0.008 mmHg respectively.
Pooled factual audit: {"n_stays": 400, "n_readings": 18910, "original_nll": 1.503084563722898, "edited_nll": 1.503016271813855, "delta_nll": -6.829190904283489e-05, "stay_cluster_se": 2.799329639474757e-05, "approximate_95pct_upper_delta": -1.3425048109129657e-05, "allowed_delta": 0.005}
Training-only evidence: rich-history cross-fits, late-period analyses, multiple horizons, independent sensors, and insulin policy-threshold checks favor negligible dose-magnitude effects. Isolated nonlinear subset signals were not stable across broader samples/specifications; they were not extrapolated to doubled doses. This supports a null magnitude repair, not a claim that observational records prove absence of every possible drug effect. Drug identity/presence and measurement-value channels remain intact. No model gradients or weight optimizer used. Expected submitted readouts: all four zero.
