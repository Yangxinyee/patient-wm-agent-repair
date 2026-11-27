# Weight-repair notebook

## Objective and constraints
- Infer this simulated ICU's drug effects from the provided records only.
- Repair the four dose-rollout readouts while increasing factual value NLL by no more than 0.005 nats per reading.
- No model gradients or weight optimiser; statistical fits and computed weight edits only.
- Limits: 300 Python runs and 50 forward requests.

## Run 1
Requested architecture, data schema, vocabulary, and tensor shapes. No causal conclusions or weight edits yet.

## Run 2
Architecture: actions have a separate DeepSets encoder, then enter the posterior and recurrent prelude. Reading heads also have same-code last-reading registers and transition logits. No drug effects inferred yet. Loaded train/tuning records and inspected schedule and drug encodings.

## Run 3
Built complete hourly panel (-1..24); an absent infusion is zero rate. Readings are at integer hours, infusions at hour + 0.0833. Compared next-primary-reading regressions with increasingly rich pre-treatment controls; standard errors clustered by stay.
{
  "insulin": {
    "unadjusted": {
      "effect_per_rate_unit": 23.02625,
      "cluster_SE": 0.09477,
      "residual_dose_SD": 1.08013,
      "dose_R2": 0.0,
      "outcome_residual_SD": 24.864,
      "n": 280327
    },
    "current + time": {
      "effect_per_rate_unit": -1.96305,
      "cluster_SE": 0.04693,
      "residual_dose_SD": 0.63989,
      "dose_R2": 0.649,
      "outcome_residual_SD": 14.117,
      "n": 280327
    },
    "+ reading history": {
      "effect_per_rate_unit": -1.9886,
      "cluster_SE": 0.04175,
      "residual_dose_SD": 0.63988,
      "dose_R2": 0.649,
      "outcome_residual_SD": 12.78,
      "n": 280327
    },
    "+ assays, other drug, past doses": {
      "effect_per_rate_unit": -1.93093,
      "cluster_SE": 0.04083,
      "residual_dose_SD": 0.63898,
      "dose_R2": 0.65,
      "outcome_residual_SD": 12.512,
      "n": 280327
    }
  },
  "norepinephrine": {
    "unadjusted": {
      "effect_per_rate_unit": -30.00927,
      "cluster_SE": 0.21648,
      "residual_dose_SD": 0.34538,
      "dose_R2": 0.0,
      "outcome_residual_SD": 10.384,
      "n": 298808
    },
    "current + time": {
      "effect_per_rate_unit": -2.22168,
      "cluster_SE": 0.04382,
      "residual_dose_SD": 0.23558,
      "dose_R2": 0.5348,
      "outcome_residual_SD": 4.839,
      "n": 298808
    },
    "+ reading history": {
      "effect_per_rate_unit": 0.2547,
      "cluster_SE": 0.03553,
      "residual_dose_SD": 0.22603,
      "dose_R2": 0.5717,
      "outcome_residual_SD": 4.389,
      "n": 298808
    },
    "+ assays, other drug, past doses": {
      "effect_per_rate_unit": 2.40061,
      "cluster_SE": 0.08922,
      "residual_dose_SD": 0.09671,
      "dose_R2": 0.9216,
      "outcome_residual_SD": 4.35,
      "n": 298808
    }
  }
}
These preliminary regressions are diagnostic, not yet accepted as causal. Next: baseline model forward results and nonlinear/history robustness checks.

## Forward requests 1 and 2
Original factual NLL (tuning IDs 2020000000..99): 1.63461; provisional maximum 1.63961.
Original rollout: insulin x2 -1.847, half +0.777; NE x2 -0.498, half -0.600.

## Run 4
Cross-fitted nonlinear nuisance regressions, split by stay, active infusions only. Features use current/past readings, observation ages, online smoothing, pre-ICU baseline, prior doses, time and other drug. Estimated residual-dose slopes and sensitivity across dose/hour subsets.
{
  "insulin": {
    "all active": [
      -1.99285,
      0.04554,
      0.59887,
      253863
    ],
    "hours >=12": [
      -2.04868,
      0.05966,
      0.61587,
      142454
    ],
    "lower half of doses": [
      -1.94372,
      0.07221,
      0.40383,
      158100
    ],
    "upper half of doses": [
      -2.01779,
      0.05771,
      0.73907,
      95763
    ],
    "away from maximum": [
      -1.99092,
      0.04556,
      0.59868,
      253862
    ],
    "current primary observed": [
      -1.99783,
      0.0459,
      0.60385,
      246971
    ]
  },
  "norepinephrine": {
    "all active": [
      3.29011,
      0.09111,
      0.13328,
      146556
    ],
    "hours >=12": [
      3.38311,
      0.11624,
      0.13967,
      75241
    ],
    "lower half of doses": [
      3.50921,
      0.15079,
      0.10196,
      74248
    ],
    "upper half of doses": [
      3.17119,
      0.11255,
      0.14585,
      72308
    ],
    "away from maximum": [
      3.29965,
      0.10114,
      0.1253,
      124482
    ],
    "current primary observed": [
      3.29235,
      0.09268,
      0.13325,
      139375
    ]
  }
}

## Run 5
Estimated 1..12 hour local projections using held-out history models. Subtracted subsequent-treatment feedback using future-dose residual covariances to estimate controlled impulse responses. Geometric summaries are tentative; long-lag selection and recursive estimation can amplify error. No weight edits yet.
{
  "insulin": {
    "rows": [
      {
        "lag": 1,
        "total_LP": -1.9915,
        "controlled_impulse": -1.9915,
        "total_LP_SE": 0.04557,
        "next_dose_feedback": 0,
        "n": 253863
      },
      {
        "lag": 2,
        "total_LP": -2.2912,
        "controlled_impulse": -2.3867,
        "total_LP_SE": 0.05054,
        "next_dose_feedback": -0.0479,
        "n": 227223
      },
      {
        "lag": 3,
        "total_LP": -1.8451,
        "controlled_impulse": -2.0916,
        "total_LP_SE": 0.05563,
        "next_dose_feedback": -0.0497,
        "n": 210298
      },
      {
        "lag": 4,
        "total_LP": -1.3483,
        "controlled_impulse": -1.7057,
        "total_LP_SE": 0.05969,
        "next_dose_feedback": -0.0503,
        "n": 196607
      },
      {
        "lag": 5,
        "total_LP": -0.772,
        "controlled_impulse": -1.1955,
        "total_LP_SE": 0.06484,
        "next_dose_feedback": -0.0503,
        "n": 184608
      },
      {
        "lag": 6,
        "total_LP": -0.4743,
        "controlled_impulse": -0.9007,
        "total_LP_SE": 0.069,
        "next_dose_feedback": -0.0493,
        "n": 173365
      },
      {
        "lag": 7,
        "total_LP": -0.2797,
        "controlled_impulse": -0.6669,
        "total_LP_SE": 0.07385,
        "next_dose_feedback": -0.0497,
        "n": 162488
      },
      {
        "lag": 8,
        "total_LP": -0.1758,
        "controlled_impulse": -0.5084,
        "total_LP_SE": 0.07888,
        "next_dose_feedback": -0.0488,
        "n": 151796
      },
      {
        "lag": 9,
        "total_LP": -0.0703,
        "controlled_impulse": -0.3329,
        "total_LP_SE": 0.08706,
        "next_dose_feedback": -0.0504,
        "n": 141116
      },
      {
        "lag": 10,
        "total_LP": 0.064,
        "controlled_impulse": -0.1455,
        "total_LP_SE": 0.09255,
        "next_dose_feedback": -0.0499,
        "n": 130635
      },
      {
        "lag": 11,
        "total_LP": -0.011,
        "controlled_impulse": -0.1475,
        "total_LP_SE": 0.10163,
        "next_dose_feedback": -0.0489,
        "n": 120066
      },
      {
        "lag": 12,
        "total_LP": 0.0378,
        "controlled_impulse": -0.0541,
        "total_LP_SE": 0.11201,
        "next_dose_feedback": -0.0491,
        "n": 109636
      }
    ],
    "geometric": [
      -2.447715382968271,
      0.8378827397427017
    ],
    "kernel": [
      -1.9915089827921182,
      -2.3866989028456365,
      -2.0915635810491975,
      -1.7057206233141478,
      -1.1955081728687924,
      -0.9007079162430982,
      -0.666932789618994,
      -0.5084256062282179,
      -0.33290715412720795,
      -0.145537396665059,
      -0.14746801051036365,
      -0.05411996528443585
    ]
  },
  "norepinephrine": {
    "rows": [
      {
        "lag": 1,
        "total_LP": 3.2932,
        "controlled_impulse": 3.2932,
        "total_LP_SE": 0.09112,
        "next_dose_feedback": 0,
        "n": 146556
      },
      {
        "lag": 2,
        "total_LP": 4.9976,
        "controlled_impulse": 2.5879,
        "total_LP_SE": 0.11116,
        "next_dose_feedback": 0.7317,
        "n": 133445
      },
      {
        "lag": 3,
        "total_LP": 5.5015,
        "controlled_impulse": 1.7747,
        "total_LP_SE": 0.12409,
        "next_dose_feedback": 0.7315,
        "n": 127569
      },
      {
        "lag": 4,
        "total_LP": 5.2038,
        "controlled_impulse": 0.9874,
        "total_LP_SE": 0.13706,
        "next_dose_feedback": 0.736,
        "n": 121667
      },
      {
        "lag": 5,
        "total_LP": 4.481,
        "controlled_impulse": 0.4273,
        "total_LP_SE": 0.14982,
        "next_dose_feedback": 0.7373,
        "n": 115815
      },
      {
        "lag": 6,
        "total_LP": 3.8049,
        "controlled_impulse": 0.2757,
        "total_LP_SE": 0.15714,
        "next_dose_feedback": 0.7396,
        "n": 109912
      },
      {
        "lag": 7,
        "total_LP": 3.0577,
        "controlled_impulse": 0.0074,
        "total_LP_SE": 0.16998,
        "next_dose_feedback": 0.7415,
        "n": 103989
      },
      {
        "lag": 8,
        "total_LP": 2.5549,
        "controlled_impulse": 0.0318,
        "total_LP_SE": 0.1813,
        "next_dose_feedback": 0.7467,
        "n": 98005
      },
      {
        "lag": 9,
        "total_LP": 2.246,
        "controlled_impulse": 0.1012,
        "total_LP_SE": 0.19735,
        "next_dose_feedback": 0.7482,
        "n": 92036
      },
      {
        "lag": 10,
        "total_LP": 1.8218,
        "controlled_impulse": -0.0967,
        "total_LP_SE": 0.21019,
        "next_dose_feedback": 0.7529,
        "n": 85957
      },
      {
        "lag": 11,
        "total_LP": 1.6283,
        "controlled_impulse": 0.0394,
        "total_LP_SE": 0.23231,
        "next_dose_feedback": 0.754,
        "n": 79945
      },
      {
        "lag": 12,
        "total_LP": 1.4078,
        "controlled_impulse": -0.0408,
        "total_LP_SE": 0.25115,
        "next_dose_feedback": 0.7563,
        "n": 73880
      }
    ],
    "geometric": [
      3.4251262905273707,
      0.6730035660080467
    ],
    "kernel": [
      3.293228277726093,
      2.5878650009977466,
      1.7746983768126912,
      0.9873506803028125,
      0.4273030715972208,
      0.2756797388201844,
      0.007392273242067926,
      0.03180661616606528,
      0.10121714894504308,
      -0.09665213231440539,
      0.03943189864870522,
      -0.04081733879812144
    ]
  }
}

## Run 7
Recovered architecture. Separate action encoder feeds posterior and prelude; reading heads also use same-code last-reading registers and transition logits. No edits. Run 6 record-based rollout targets remain provisional: insulin x2 about -14.18, half +7.09; norepinephrine x2 +4.68, half -2.34 on training schedules.

## Run 8
Recovered persistent analysis variables and exact tensor shapes. Value-bin head is Linear(192,192), activation, Linear(192,17). Action encoder is separate two-stage DeepSets MLP. Last_numeric has 8 inputs; exact reconstruction must be checked rather than assuming the architecture summary lists every feature.

## Forward 4
Original factual predictions on the 100 training stays with saved states: NLL 1.63841 over 4510 readings. By event: finger 1.4350, lab 1.3445, arterial 1.5024, cuff 1.9645. No edits. Intended for validating a local reconstruction of the reading head.

## Run 10
Loaded original tensors and matched training factual CSV schema. Established complete numeric normalisation and fine-bin tables. Preparing a forward-head reconstruction; no model edits or external information used.

## Run 11
Matched 4510 factual reading rows to their preceding saved states. Initial local head reconstruction identifies measurement type index 1 but mean RMSE is about 3.76, too large to assume exact reconstruction. Checking bin boundaries, register layout, status and latent details before editing.

## Run 12
Register search supports [has_last, age, z_value, has_value, ...], measurement type 1 and zero status. Bin-edge convention has negligible effect here. Residual reconstruction error remains ~3.76; checking mixed MLP activations and latent Monte Carlo noise next. No edits.

## Restart / Run 14
No edits. Head reconstruction error exceeds four-draw Monte Carlo noise; recovering architecture, persistent variables, and baseline rollout results before attempting a causal weight edit.

## Run 15
Original rollout on the initial tuning sample: reached-hour means insulin x2 -1.85, half +0.78; norepinephrine x2 -0.50, half -0.60. These are substantially weaker than controlled record-based estimates (and NE x2 has the wrong sign). Still no edits.

## Runs 16–19
State alignment and out=norm(mean h) verified. Status embedding for readings is zero. Separate gap conventions did not resolve local head error (~3.36 versus ~0.93 four-draw noise RMS). Testing register age at saved-state time versus forecast time, type index, and numeric layout. No edits.

## Run 20: compensated continuous-dose trial
Exact head reconstruction still unverified (best deterministic RMS 2.67). Testing gamma=4 on numeric dose/rate columns with per-bin mean compensation in dose/rate embedding tables. This preserves pooled mean factual atom embeddings while amplifying within-bin and out-of-range dose variation. Not yet validated; will reject if factual NLL exceeds the budget.

## Forward 5–6: rejected continuous-dose amplification
gamma=4 with pooled bin compensation: factual tuning NLL 1.63458 vs original 1.63461. Rollouts insulin x2 -2.051, half +1.116; NE x2 -1.140, half -0.600. Insulin improvement too small and NE x2 worsened. Reverting all three embedding edits.

## Run 25: systematic local-head check
Best deterministic reconstruction: (0.0581074059009552, 1.68510901927948, ('flow', 'seconds_div10', 'hours_div10', 'gelu', 'gelu'), (6, 3, 'forecast_hours', 'gelu')). No weight edits. Must still compare to latent sampling noise before trusting a head edit.

## Run 26: latent integration check
[{"config": ["flow", "hours_div10", "gelu", "split", [6, 3, "forecast_hours"]], "mean_rmse": 1.6213697818927704, "expected_four_draw_rmse": 1.4728203234352455, "normalized_rmse": 0.05880779470105488, "normalized_noise": 0.06145674633075733, "excess_normalized_mse": -0.00031857495196163654, "mean_log_sd": -0.39646247029304504}, {"config": ["no_flow", "hours_div10", "gelu", "split", [6, 3, "forecast_hours"]], "mean_rmse": 1.8476455397829779, "expected_four_draw_rmse": 2.037750406616455, "normalized_rmse": 0.06510749887839799, "normalized_noise": 0.09084452029591543, "excess_normalized_mse": -0.004013740457594395, "mean_log_sd": -0.09486968070268631}, {"config": ["flow", "seconds_div10", "gelu", "split", [6, 3, "forecast_hours"]], "mean_rmse": 1.696447528647144, "expected_four_draw_rmse": 1.3953982097277995, "normalized_rmse": 0.058303355794319875, "normalized_noise": 0.05360864956275166, "excess_normalized_mse": 0.0005253939889371395, "mean_log_sd": -0.4706123173236847}, {"config": ["no_flow", "hours", "gelu", "split", [6, 3, "forecast_hours"]], "mean_rmse": 1.6672157892975, "expected_four_draw_rmse": 1.164809480159165, "normalized_rmse": 0.059641018320525124, "normalized_noise": 0.04333269220780178, "excess_normalized_mse": 0.0016793288523331285, "mean_log_sd": -0.5911003947257996}, {"config": ["no_flow", "seconds_div10", "gelu", "split", [6, 3, "forecast_hours"]], "mean_rmse": 1.6861378152119766, "expected_four_draw_rmse": 1.259079147308344, "normalized_rmse": 0.059476099791531435, "normalized_noise": 0.046408550244777455, "excess_normalized_mse": 0.0013836529105901718, "mean_log_sd": -0.5083918571472168}, {"config": ["no_flow", "hours_div10", "silu", "split", [6, 3, "forecast_hours"]], "mean_rmse": 1.677154760636444, "expected_four_draw_rmse": 1.6695922526245155, "normalized_rmse": 0.06289189008660595, "normalized_noise": 0.07568763906681096, "excess_normalized_mse": -0.001773228868842125, "mean_log_sd": -0.40057265758514404}]
Original weights remain unchanged. Next: use dose innovations and paired factual residuals to construct a small statistical head correction, rather than amplify the existing dose direction.

## Runs 27–31: recovery and head-instrument diagnostics
Recovered architecture, aligned states and DML caches. No edits. Building antithetic latent-averaged reading-head features and measuring residual-dose response plus held-out innovation decodability.

## Run 34: restart recovery
Run 33 verified exact state packing. Last-bin boundaries disagree in 163 cached rows; prior and reading-view gap scalings differ. Recovering architecture and causal caches before any weight edit.

## Runs 35–38: aligned residual and decoder checks
Recovered 4510 antithetic-averaged head features. On the initial 100 training stays, factual observed-minus-predicted slopes against dose innovations are insulin -1.857 and NE +3.833. These are small-sample diagnostics, not new causal estimates. Checking outcome-cache alignment and cross-stay innovation decoding. No weights edited.

## Run 38 result / Run 39
Confirmed DML outcomes are next-reading minus last-reading, not levels. Head-feature innovation decoders failed cross-stay validation (best all-row R2 insulin -0.003, NE -0.036). Rejecting that head-only edit. Testing earlier state/posterior signals and recovering the explicit action pathway instead. No current edits.

## Run 39 result / Run 40
Insulin dose innovation has held-out R2 about 0.27 in out/mean_h, 0.40 in posterior z, and 0.37 in all slots (active primary rows). NE innovation remains weak (posterior z R2 0.058; out negative). Testing whether the earlier head decoder failed because it required the same linear readout to be zero for other codes.

## Run 41: pathway recovery
Run 40 found insulin innovation decodable from the reading query (held-out R2 0.313), but not from head-hidden features; NE remained weak. No edits. Recovering architecture and causal estimates to select an earlier, controllable pathway, with factual NLL validation required.

## Run 46: register-invariant decoding
{'insulin': {'query': (0.001, 0.29490138445996406, 0.5431895253792158), 'query_no_register_axes': (0.001, 0.29002820697566745, 0.5385983094074785), 'with_code': (100.0, -0.00871122726761464, -0.02457488716355624), 'with_code_no_register_axes': (0.001, 0.290028161906869, 0.5385982669791561)}, 'norepinephrine': {'query': (10.0, 0.004805775453950112, 0.06951803981873954), 'query_no_register_axes': (10.0, 0.004925958126326013, 0.07047793165682362), 'with_code': (100.0, 0.0017492157730133462, 0.06696235405302925), 'with_code_no_register_axes': (10.0, 0.0049259596004407324, 0.07047794264625033)}}
No edits. Tested whether last-reading input directions explain the loss of innovation decodability in the head.

## Restart / Run 48
No edits. Perfect reading-type gate, but insulin innovation decoder has only 0.029 held-out R2 when inactive hours are included. Recovering the full architecture and established causal evidence before selecting an edit pathway.

## Run 53: first gated head trial
Active insulin innovation decoder held-out R2=0.3765, but ungated inactive RMS=7.993. Infusion-gated all-hour R2=0.3684; inactive RMS=0.037. Replaced two lowest-contribution value-head neurons [109, 107] with a glucose/active-insulin gated linear innovation correction; slope=-1.9929, tilt=-0.11720. Four head tensors provisionally edited. Must validate actual factual NLL and dose rollouts before retaining; no model gradients used.

## Forward 7 / Run 55: rejected gated head edit
Paired tuning factual NLL: original 1.63461, edited 38.59464. All four reading codes were damaged, so the local gating/reconstruction checks were insufficient. Reverted every edited tensor immediately. Do not retain or merely rescale this unvalidated construction.

## Run 57: rejected-head projection audit
Checking whether register-projected decoder features were mistakenly installed as coefficients on raw with-code inputs. No edits installed.

## Runs 59–60: corrected latent audit
Run 59 stopped at an incorrect tensor prefix; z_head is top-level. Run 60 tests individual-draw sensitivity and the type-embedding mismatch. No weights edited.

## Restart / Run 61
No edits. Run 60 showed severe individual-latent sensitivity in the rejected head construction, with unreliable local reconstruction. Recovering the architecture, causal estimates, and persistent helpers before another edit.

## Runs 65–68: fresh head reconstruction
Cached weights equal the original files. The prior and forecast gap inputs must be tested separately; a seconds-based /10 forecast gap sharply reduces the previous glucose reconstruction error. Rebuilding latent-aware innovation decoders and checking cross-stay performance before any new edit. No edits.

## Runs 74–79: restart audit
No edits. Recovered record-based immediate effects: insulin about -1.99 per rate unit; norepinephrine about +3.29. Provisional controlled-kernel rollout targets: insulin x2 -14.18, half +7.09; NE x2 +4.68, half -2.34. Stored reading gate is excellent on its fitted projected features but fails on sampled raw head inputs. Auditing feature projection and reconstruction before any new installation.

## Run 83: deterministic reading-head diagnostic
All previous edits were absent. Temporarily zeroed z_head.weight only, to remove prior-draw uncertainty from factual reading predictions on the 100 training stays with saved original states. This is a diagnostic, not a proposed final repair. Require near-exact local/GPU head agreement before installing another gated correction; restore z_head afterward.

## Runs 85–86: identifiable register probe
Deterministic head scan did not achieve exact agreement: best normalized mean RMSE 0.02094 and NLL RMSE 0.10588. Temporarily zeroed query_norm and value transitions, and installed an analytically constructed linear value-head probe. Its scalar register coefficients are powers of two /128, so observed probe means can identify the actual eight-input register layout. These edits are diagnostics only and must all be reverted.

## Run 87: restart recovery
Forward 9 was an artificial register probe (NLL 2.62934), not a candidate repair. Saved its diagnostic weights in r87_probe_weights and reverted every edit. Recovering the exact head layout from its saved outputs before further edits.

## Runs 88–90: register probe decoding
Original weights restored. Probe numeric coefficients verified as [1,2,4,8,16,32,64,128]/128. Inverting saved probe means to test eight-field register layouts, age units, status, and bin boundaries.

## Run 96: probe-resolved reading register
Controlled probe identifies numeric fields 0=presence, 3=log1p(age_seconds)/10, 6=last normalized value; other reading fields zero, left-sided bins, measurement status zero. Testing deterministic reconstruction against forward 8. No edits.

## Run 97: exact deterministic head recovered
Validated all cached forward-8 readings: seconds-based time flow, gap(log(dt_seconds)/10), GELU, type embedding 1, and probe-resolved register. This removes the reconstruction uncertainty that invalidated earlier gated edits. Original weights remain unchanged.

## Run 98: stochastic head reconstruction
Testing latent prior activation/packing against original factual outputs, allowing for the GPU's four prior draws. Recovered aligned innovation targets. No weights edited.
{'activation': 'gelu', 'packing': 'split', 'normalized_MSE': 0.0028543598018586636, 'expected_four_draw_MSE': 0.0029637326952069998, 'excess_MSE': -0.00010937289334833622, 'NLL_RMSE': 0.20606064796447754, 'mean_log_sd': -0.48285219073295593}
{'activation': 'gelu', 'packing': 'interleaved', 'normalized_MSE': 0.0041332063265144825, 'expected_four_draw_MSE': 0.0040336293168365955, 'excess_MSE': 9.957700967788696e-05, 'NLL_RMSE': 0.24339871108531952, 'mean_log_sd': -0.2101394534111023}
{'activation': 'silu', 'packing': 'split', 'normalized_MSE': 0.00806079525500536, 'expected_four_draw_MSE': 0.0016576317138969898, 'excess_MSE': 0.00640316354110837, 'NLL_RMSE': 0.3298589587211609, 'mean_log_sd': -1.2057170867919922}
{'activation': 'silu', 'packing': 'interleaved', 'normalized_MSE': 0.005343444179743528, 'expected_four_draw_MSE': 0.006770724896341562, 'excess_MSE': -0.001427280716598034, 'NLL_RMSE': 0.31665846705436707, 'mean_log_sd': -0.4464951455593109}

## Run 99: latent-aware, cross-stay decoder tests
Physical active-infusion labels recovered directly from records. Fitting innovation regressions and family/infusion gates on individual prior draws; holding out 25% of stays. No weight edits.
[('insulin', (0.44562312272344906, 0.44570992471108317, 'raw', 0.1)), ('norepinephrine', (-0.04744019540362765, -0.04725591583177757, 'register_free', 10000.0))]

## Restart / Run 100
Original weights remain unchanged. Run 99 established insulin innovation held-out single-draw R2=0.446, but no useful NE decoder. Recovering exact head helpers, gates and causal estimates before an insulin-only controlled edit. Actual paired factual NLL and dose rollouts are required validation.

## Run 103: candidate preparation
Recovered stable DML immediate slopes: insulin -1.99285 (SE 0.04554), NE +3.29011 (SE 0.09111). Insulin state innovation decoder is useful; NE decoder is not. Checking standardized-feature conversion and exact cached value-head evaluation, then selecting low-contribution neurons for an insulin-only gated correction. No edits installed.

## Run 104: local gated-insulin candidate
Original tuning rollout: insulin x2 -1.847, half +0.777; NE x2 -0.498, half -0.600. Provisional targets on those exact hours: insulin -13.599/+6.800, NE +4.746/-2.373. Constructed a raw-coordinate, latent-aware insulin gate/innovation correction using neurons 107/109; assessing local held-out NLL before installing any weights.

## Run 105 result
Installation aborted before save_edit: max logit reconstruction error 0.003045 exceeded numerical threshold 0.003. No edits saved. Likely cancellation between paired gated neurons; increasing their decoder separation to reduce numerical sensitivity.

## Run 106: insulin candidate installed
Increased symmetric decoder separation from 0.05 to 0.20 to reduce cancellation. Local checks passed; full-strength gated correction installed in value-head neurons 107/109, with mean compensation. Four value-head tensors edited. Awaiting paired GPU factual validation; not yet accepted.

## Forward 10–11: first validated insulin candidate
Four value-head tensors, hash 70c11c43ec. Paired tuning factual NLL:
original 1.63461, edited 1.63592, delta +0.00131 (4751 readings).
Dose shifts on the same 100 stays:
original insulin -1.847/+0.777; edited -5.908/+2.782.
Original NE -0.498/-0.600; edited -0.517/-0.615.
Insulin intervention transmission improved substantially, but remains below
provisional record-derived targets -13.599/+6.800. NE is not repaired.
Backed up this factual-budget-compliant candidate in memory and
/work/insulin_candidate_v1.npz before any further changes.

## Runs 108–110: causal and construction recovery
Immediate residual-dose effects are stable across hour/dose strata:
insulin approximately -1.99 per rate unit; NE approximately +3.29.
Cumulative targets use controlled impulse kernels and remain provisional.
Recovered exact local amplitude evaluator and existing gated construction;
testing scaling analytically before another weight installation.

## Run 112: provisional insulin strength 2
Scaled only the two correction output columns; gate, decoder, and
removed-neuron mean compensation are unchanged.
Local delta NLL at strength 2: all +0.002618, training +0.001343,
held-out +0.006454. Held-out result is a generalization warning.
This is a trial, not an accepted replacement. Saved strength-1 candidate
remains available for immediate restoration. GPU factual check is next.

## Restart / Runs 113–115 and Forward 12–13
Strength-2 insulin candidate (hash 43d7176a64): paired factual NLL on tuning stays 2020000100..199 is 1.62310 vs 1.62044, delta +0.00266. On tuning stays 2020000000..99 its dose shifts are insulin x2 -10.248, half +5.053; NE x2 -0.513, half -0.604. Provisional record-derived targets for those rollout hours remain -13.599/+6.800 and +4.746/-2.373. Saved current strength-2 tensors in r115_strength2. Further amplification requires factual-budget validation; NE remains unrepaired.

## Run 116: insulin strength-2.6 trial
Installed strength 2.6 by scaling only value-head output columns 107/109 of the saved strength-1 construction. Gate, decoder, and removed-neuron mean compensation are unchanged. Strength-2 fallback remains in r115_strength2. This trial is not accepted pending a fresh paired factual check.

## Forward 14 / Run 117: strength 2.6 rejected provisionally
On fresh tuning stays 2020000200..299, NLL is 1.62614 edited vs 1.61959 original, delta +0.00655. Restored strength-2 insulin fallback. Exploring controlled action-embedding state contrasts for a lower-factual-impact NE correction, rather than amplifying the weak observational NE innovation decoder.

## Run 124: controlled NE representation probe
Verified exact head-input reconstruction against cached original states. Temporarily shifted only NE embed.code by the continuous numeric direction corresponding to +0.10 rate. This does not change bins and may also affect END atoms, so it is a diagnostic direction, not the actual dose_edit intervention. Must restore embed.code immediately after paired states are obtained.

## Run 125: probe recovery
Removed the diagnostic embed.code.weight edit immediately after forward 15. The four strength-2 insulin value-head edits remain. Loaded aligned original/probed states to investigate a controlled NE response direction; no NE repair has been accepted.

## Runs 129–130: controlled NE head-direction test
Run 129 stopped on the incorrect core.z_prior prefix; no edits. Run 130 register consistency error 4.1671991. Testing constrained directions on held-out stays; no new edits.

## Run 131: latent-cache mismatch audit
Best query cache: cached qq98; register draw-invariance error 1.6689301e-06. No new weights installed. Rebuilding paired inputs with one consistent prior convention rather than mixing cached latent conventions.

## Run 132: verified NE direction efficiency
Rebuilt inputs agree with the selected gelu/split prior cache. Computed covariance-constrained NE probe directions using training stays, evaluated their factual variation and probe response on held-out stays. Results:
[{"ridge": 1e-07, "train_SD_per_unit_probe_response": 2.1483134156131762, "validation_factual_mean_SD": [0.37868564838024527, 2.7753602359121503], "validation_probe_response_mean_SD": [1.0786586666046272, 0.6391174579632463], "factual_MAP_SD_if_probe_calibrated_to_0.329": 0.9131240465776925, "weight_norm": 1372.9319605940466}, {"ridge": 1e-06, "train_SD_per_unit_probe_response": 2.2441324513041296, "validation_factual_mean_SD": [0.4668558966372616, 2.941045116660891], "validation_probe_response_mean_SD": [1.0786428274645243, 0.619494147095025], "factual_MAP_SD_if_probe_calibrated_to_0.329": 0.9676361948777165, "weight_norm": 910.4446430319636}, {"ridge": 1e-05, "train_SD_per_unit_probe_response": 2.642791498007302, "validation_factual_mean_SD": [0.47472113210175787, 3.36510643744316], "validation_probe_response_mean_SD": [1.104674017862256, 0.5962262146799052], "factual_MAP_SD_if_probe_calibrated_to_0.329": 1.1071570340896115, "weight_norm": 451.8868103564695}, {"ridge": 0.0001, "train_SD_per_unit_probe_response": 3.3846197671557574, "validation_factual_mean_SD": [0.2705246688909235, 4.235324779645371], "validation_probe_response_mean_SD": [1.1458081178308919, 0.5682093060675222], "factual_MAP_SD_if_probe_calibrated_to_0.329": 1.393468441075903, "weight_norm": 230.49478676043864}, {"ridge": 0.001, "train_SD_per_unit_probe_response": 4.796290083676495, "validation_factual_mean_SD": [0.20451191301663332, 5.520355684869838], "validation_probe_response_mean_SD": [1.1706060341007631, 0.5496445022100264], "factual_MAP_SD_if_probe_calibrated_to_0.329": 1.81625774423471, "weight_norm": 126.59697219438657}, {"ridge": 0.01, "train_SD_per_unit_probe_response": 7.494145521275962, "validation_factual_mean_SD": [0.5359714159985296, 7.896912788083896], "validation_probe_response_mean_SD": [1.1722724758630239, 0.48231591599648643], "factual_MAP_SD_if_probe_calibrated_to_0.329": 2.598171173320271, "weight_norm": 72.56227832954471}]
Probe calibration to 0.329 is diagnostic, not a true dose intervention: the code perturbation also affects END atoms and leaves bins unchanged. No new weights installed.

## Runs 133–136: restart recovery
Architecture and causal checks recovered. Existing strength-2 insulin edit remains unchanged. NE immediate adjusted slope +3.29011 is stable across dose/hour strata; the artificial code probe is only a representation diagnostic, not a causal intervention. Recovering exact cached head probabilities and reading labels before a gated NE local-loss test.

## Runs 137–138: verified local NE scan
Cached labels/probabilities verified exactly. Strength-2 insulin local NLL delta is +0.002618 overall, +0.001343 on 75 fit stays and +0.006453 on 25 held-out stays. Selected low-contribution neurons [48, 171]. Scanning covariance-constrained NE directions with the actual paired-GELU gate. Probe shifts are diagnostics, not true dose-effect targets. No new edits installed.
[{'ridge': 1e-07, 'probe_shift': 0.1, 'scale': np.float64(0.09992041359156133), 'all': 0.002654307054567892, 'fit': 0.0015125010412416371, 'validation': 0.0060898744813317795, 'increment_MAP': 7.970173021657452e-05, 'increment_glucose': -2.7211369218814174e-05}, {'ridge': 1e-07, 'probe_shift': 0.2, 'scale': np.float64(0.19984082718312265), 'all': 0.0029493281127730576, 'fit': 0.001877964587039841, 'validation': 0.006172941921312557, 'increment_MAP': 0.0005749981714407499, 'increment_glucose': -2.711338052519669e-05}, {'ridge': 1e-07, 'probe_shift': 0.329011, 'scale': np.float64(0.3287491519617318), 'all': 0.003724122220778059, 'fit': 0.0026522521424537167, 'validation': 0.006949260189780633, 'increment_MAP': 0.001875850185630229, 'increment_glucose': -2.6985684807204473e-05}, {'ridge': 1e-06, 'probe_shift': 0.1, 'scale': np.float64(0.0999165895390723), 'all': 0.0025641521087736976, 'fit': 0.0014462322946115114, 'validation': 0.005927848616275031, 'increment_MAP': -8.922157611379383e-05, 'increment_glucose': -1.3729946181829214e-06}, {'ridge': 1e-06, 'probe_shift': 0.2, 'scale': np.float64(0.1998331790781446), 'all': 0.0028147479842321557, 'fit': 0.001778167856393122, 'validation': 0.005933702413330051, 'increment_MAP': 0.0003119636947654149, 'increment_glucose': 2.746678648713099e-05}, {'ridge': 1e-06, 'probe_shift': 0.329011, 'scale': np.float64(0.32873657040839716), 'all': 0.0036003063028155896, 'fit': 0.002558028413718768, 'validation': 0.0067364046624535795, 'increment_MAP': 0.00160302905687543, 'increment_glucose': 6.862164075455652e-05}, {'ridge': 1e-05, 'probe_shift': 0.1, 'scale': np.float64(0.09986840237599656), 'all': 0.0023470435177655215, 'fit': 0.0012309636769726824, 'validation': 0.005705203749839974, 'increment_MAP': -0.0004361513618897608, 'increment_glucose': -2.7308488506684794e-05}, {'ridge': 1e-05, 'probe_shift': 0.2, 'scale': np.float64(0.19973680475199312), 'all': 0.002482408421199294, 'fit': 0.0014446711575605371, 'validation': 0.005604844543347906, 'increment_MAP': -0.00020886330735404447, 'increment_glucose': -2.730848850483272e-05}, {'ridge': 1e-05, 'probe_shift': 0.329011, 'scale': np.float64(0.32857802934129005), 'all': 0.0032685647453330863, 'fit': 0.002216203753577848, 'validation': 0.006435002040525515, 'increment_MAP': 0.0011111534543135084, 'increment_glucose': -2.730848850244318e-05}, {'ridge': 0.0001, 'probe_shift': 0.1, 'scale': np.float64(0.09981576632002546), 'all': 0.0021856292239820028, 'fit': 0.00097770812410378, 'validation': 0.005820129600060031, 'increment_MAP': -0.0007071783406537576, 'increment_glucose': -2.730848850864436e-05}, {'ridge': 0.0001, 'probe_shift': 0.2, 'scale': np.float64(0.19963153264005093), 'all': 0.002339622868126506, 'fit': 0.0011117754548574826, 'validation': 0.006034079307162633, 'increment_MAP': -0.0004486112017513356, 'increment_glucose': -2.7308488508751852e-05}, {'ridge': 0.0001, 'probe_shift': 0.329011, 'scale': np.float64(0.32840485092717897), 'all': 0.0034140970649823407, 'fit': 0.0020374303117126073, 'validation': 0.007556334362598382, 'increment_MAP': 0.001355513380460157, 'increment_glucose': -2.7308488508890874e-05}, {'ridge': 0.001, 'probe_shift': 0.1, 'scale': np.float64(0.09980746374216955), 'all': 0.002063928535650217, 'fit': 0.0007388061687485838, 'validation': 0.006051074501838686, 'increment_MAP': -0.0009115231138909741, 'increment_glucose': -2.730851233605121e-05}, {'ridge': 0.001, 'probe_shift': 0.2, 'scale': np.float64(0.1996149274843391), 'all': 0.002521266466213066, 'fit': 0.001049387750763205, 'validation': 0.006949985978922203, 'increment_MAP': -0.0001436176372342943, 'increment_glucose': -2.7308536163568472e-05}, {'ridge': 0.001, 'probe_shift': 0.329011, 'scale': np.float64(0.3283775345327494), 'all': 0.004630157463978095, 'fit': 0.002829048677719853, 'validation': 0.010049493678630672, 'increment_MAP': 0.0033973721081083197, 'increment_glucose': -2.7308566903688838e-05}, {'ridge': 0.01, 'probe_shift': 0.1, 'scale': np.float64(0.10010794449157301), 'all': 0.002534046614748818, 'fit': 0.0011115161899991506, 'validation': 0.006814282603884485, 'increment_MAP': -0.00012214975142439387, 'increment_glucose': -2.7321842846216046e-05}, {'ridge': 0.01, 'probe_shift': 0.2, 'scale': np.float64(0.20021588898314602), 'all': 0.005124585029175809, 'fit': 0.0033754166885438365, 'validation': 0.010387638214099569, 'increment_MAP': 0.004227571994101974, 'increment_glucose': -2.7335197790345542e-05}, {'ridge': 0.01, 'probe_shift': 0.329011, 'scale': np.float64(0.32936614925116925), 'all': 0.012629244386290922, 'fit': 0.010176493319071275, 'validation': 0.020009299819658485, 'increment_MAP': 0.01682848231746203, 'increment_glucose': -2.735242802935834e-05}]

## Run 139: NE trial installed
Four-head-tensor candidate bacb0c2adfe8: retained strength-2 insulin; added paired-GELU NE correction in neurons [48, 171], ridge 1e-05, diagnostic probe calibration 0.329011. Local NLL delta overall 0.003269. Strength-2 insulin-only fallback retained in r139_fallback and r115_strength2. Pending paired GPU factual and actual dose-edit validation; not yet accepted.

## Run 140: factual trial failed acceptance
Forward 16: original 1.64249, edited 1.64977 on 100 tuning stays; delta +0.00728 exceeds +0.005. Candidate not accepted. Recovering prior evidence to distinguish insulin baseline loss from incremental NE loss before selecting the next edit.

## Forward 17 / Run 143: actual NE response is partial
Current four-head-tensor trial on tuning 2020000000..99: insulin x2 -10.255, half +5.021; NE x2 +2.351, half -0.893. NE doubling now has the record-supported sign, but both NE effects remain below provisional targets +4.746/-2.373. Factual delta +0.007284 on tuning 300..399 remains unacceptable as a standalone check. Investigating factual feature leakage and decoder fitting before amplification; no new edit installed.

## Run 149: isolate factual failure
Restored r115_strength2, removing the unaccepted NE trial. Combined-trial tensors remain backed up in r149_trial_backup and r139_candidate. Next check isolates insulin factual loss on tuning 300..399, where combined delta was +0.007284.

## Forward 18 / Run 150: insulin fallback passes narrowly
On tuning 300..399, strength-2 insulin factual NLL is 1.64733 vs original 1.64249 (delta approximately +0.00484). NE trial's extra factual loss is approximately +0.00244. Need better factual efficiency before scaling either correction. Insulin fallback remains installed.

## Run 157: warm-up insulin trial
Installed strength-2.6 insulin with onset-2 gate, C=0.1, sharpness=3. Paired neurons and decoder unchanged; only their common gate and output strength changed. Exact cached local NLL delta +0.002615. Strength-2 fallback saved in r157_fallback. This is an unaccepted trial pending factual and rollout checks.

## Forward 19 / Run 158: warm-up trial rejected
On tuning 300..399, warm-up strength-2.6 NLL was 1.65674 vs 1.64249 original (delta +0.01425). Local gate performance did not generalize. Restored the strength-2 insulin fallback before analysis.

## Run 164: guarded insulin trial
Installed strength-2.6 insulin with warmup support guard (penalty 4, threshold 1), hash dfd2aed2aaef. Cached factual NLL delta 0.002025; cached held-out delta approximately +0.003182. Strength-2 fallback retained in r164_fallback and r115_strength2. This is an unaccepted trial pending GPU factual and dose-rollout checks.

## Forward 20–21 / Run 165: guarded insulin validated provisionally
On tuning 300..399, factual NLL delta approximately +0.00141. On tuning 0..99, dose shifts: insulin x2 -14.397, half +6.597; NE x2 -0.538, half -0.599. Record-derived targets on that cohort remain provisional: -13.599/+6.800 and +4.746/-2.373. Four-tensor insulin backup retained in r165_insulin_backup. Investigating NE factual efficiency; no NE edit installed yet.

## Run 170: combined guarded-insulin / prior-NE trial
Installed four-tensor candidate d98f2114563f: retained validated guarded strength-2.6 insulin and added the prior NE correction unchanged. Cached overall NLL delta 0.002675, held-out 0.003162. These are diagnostic only; candidate is unaccepted pending GPU factual and actual rollout checks. Guarded insulin fallback is preserved in r170_insulin_backup.

## Runs 171–173: restart audit and combined-trial results
Re-read architecture and record schema. Forward 22 factual on tuning 300..399: original 1.64248740, combined 1.64634215, delta +0.00385475; passes this cohort. Forward 23 on tuning 0..99: insulin x2 -14.331, half +6.691; NE x2 +2.387, half -0.873. Combined edit is a provisional fallback, not yet validated across other cohorts. NE remains weaker than provisional record-derived targets, especially for half-dose. Four-tensor backup r171_combined_backup retained. No new weight change.

## Run 179: restart after failed reconstruction assertion
Run 178 had a maximum logit reconstruction error of 1.640755 and cached NLL delta +0.013017. These diagnostics are not accepted as evidence about candidate performance until the reconstruction is audited. No weights changed in this audit run.

## Run 180: restore checked combined fallback
Restored r171_combined_backup, whose GPU factual delta was +0.00385475 on tuning 300..399 and rollout shifts were insulin -14.331/+6.691, NE +2.387/-0.873 on tuning 0..99. The failed run-178 trial is not accepted. Auditing its construction before any further trial.

## Run 182: transition-logit audit
Fresh trial reconstruction after adding the register transition term has maximum error 0.00048368365. Recomputed local diagnostics: [{'candidate': 'checked_combined', 'all': 0.0026748778764158487, 'fit': 0.0025129851419478655, 'validation': 0.003161995904520154, 'by_event': {'glucose_finger': 0.0064850980415940285, 'glucose_lab': 0.0012155240401625633, 'map_arterial': 0.0014473526971414685, 'map_cuff': 0.0009375684894621372}}, {'candidate': 'corrected_trial', 'all': 0.003386198775842786, 'fit': 0.0028193644247949123, 'validation': 0.005091740284115076, 'by_event': {'glucose_finger': 0.0064850980415940285, 'glucose_lab': 0.0012155240401625633, 'map_arterial': 0.006332510616630316, 'map_cuff': -1.9040047845919617e-05}}]. No trial installed; checked combined fallback remains active.

## Run 183: guarded stronger-NE trial installed
The run-178 discrepancy was primarily an omitted register transition-logit term, not unintended weight changes. Adding it reduces reconstruction error to 0.00048368365; direct local loss agrees with the earlier algebraic screen. Installed trial 38402104b27473b8: NE pair [48,171], arterial-only common-gate adjustment, output multiplier 1.8176594; retains validated guarded insulin correction. Local NLL delta 0.0033861988, held-out 0.0050917403. This is an UNACCEPTED trial pending paired GPU factual and dose-rollout checks. Checked combined fallback retained in c183_fallback.

## Run 184: restart audit
Forward 25 stronger-NE trial: insulin -14.423/+6.644; NE +4.736/-1.201 on tuning 0..99. Doubling matches the provisional cohort target but half-dose remains weak. No additional edit; recovering factual result and causal-estimation evidence before further changes.

## Run 185: preserve stronger trial
Forward 24 factual on tuning 300..399: 1.64524 vs 1.64249, delta about +0.00275, below +0.005. Current stronger-NE four-tensor trial preserved in c185_current; no new weight change. Broader factual validation and half-dose asymmetry remain open.

## Run 186: stronger trial fails second factual cohort
Forward 26 on tuning 100..199: original 1.62044, edited 1.62827, delta +0.00783. Stronger trial is not accepted. Its two tested cohorts pool to approximately +0.00529, leaving no safety margin. Preserved in c185_current; inspecting correction and causal persistence before revision.

## Run 187: gate and persistence audit
No weights changed. Auditing paired-neuron common gates and dose contrasts; previous cached factual checks are insufficient for acceptance after Forward 26.

## Run 188: correct event-label audit
Run 187 gate-by-event output used scalar r170_event and is invalid; recomputed with r176_events. No weights changed.

## Run 189: recover controlled-dose probes
Correct gates isolate drug/reading types, but activate on only part of the relevant readings. Investigating dose-embedding asymmetry and recovering existing controlled state-probe machinery before another edit. No weights changed.

## Run 190: NE probe limitation and loss decomposition
Recovered that the NE probe shifts embed.code continuously by +0.1-rate direction, including possible END atoms, and does not implement real dose-bin changes. Testing symmetric decoder perturbations and removed-neuron loss; no edit installed.

## Runs 191–196: restart audit and restore
Re-read architecture and record schema; recovered cross-fitted drug slopes (insulin about -2, NE about +3.3 per rate unit) and tentative persistence evidence. The prior NE probe changes a continuous code-embedding direction rather than actual dose bins; it cannot establish dose-edit symmetry. Decoder perturbations are nearly symmetric. Restored c183_fallback (checked weaker combined candidate); stronger trial preserved in c196_strong_backup. Next: validate fallback factual fit on tuning 100..199 and construct exact-dose representation probes if feasible. No new candidate accepted.

## Forward 27 / Run 197: weaker fallback fails another cohort
Checked combined fallback factual NLL on tuning 100..199: 1.62804 vs 1.62044 original, delta approximately +0.00760. Neither combined candidate is accepted across tested cohorts. Next: reconstruct an independent tuning factual-head cache from original states and validate it against GPU per-reading predictions before screening revisions.

## Run 198: restart recovery audit
Combined fallback is not accepted: factual delta +0.00760 on tuning 100–199. Recovering architecture and independent factual-cache construction before revising weights. No weights changed in this audit.

## Run 201–202: reconstruction assertion failed safely
No edits changed. The r124_x comparison failed by 2.218; auditing old-cache provenance and component differences before using local losses.

## Run 203: fresh independent tuning cache
Tuning 100–199 paired local delta 0.0073184404; GPU delta 0.0076041166. r124_x provenance is not accepted; context/prior/query reconstruction was verified. No weights changed.

## Run 207: independent candidate decomposition
      candidate      nll    delta  glucose_finger  glucose_lab  map_arterial  map_cuff
       original 1.601441 0.000000        0.000000     0.000000      0.000000  0.000000
   insulin_only 1.605140 0.003699        0.010073     0.007081      0.000028 -0.000057
  combined_weak 1.608760 0.007318        0.010055     0.007089      0.004468  0.007393
combined_strong 1.608672 0.007231        0.010055     0.007089      0.012546  0.000022
Backups converted to tensors solely for local evaluation. No weights changed.

## Run 210: generic independent cache evaluator
Completed fresh training reconstruction and built a generic exact four-head-tensor evaluator for training and tuning 100–199. It includes register transition logits and supports sparse hidden-row changes. No weights changed.

## Run 211: closed-form calibration screen
Fitted two-parameter per-reading-code probability-tilt calibrators from the fresh training cache using empirical score moments and a ridge-regularized 2x2 solve; no model gradients or weight optimizer. Screened independent tuning losses and common-gate support. No weights changed.
      candidate  calibration_scale     train   tune100
       original               0.00  0.000000  0.000000
       original               0.25 -0.000737 -0.000172
       original               0.50 -0.001317 -0.000192
       original               0.75 -0.001740 -0.000061
       original               1.00 -0.002007  0.000221
combined_strong               0.00  0.003485  0.007231
combined_strong               0.25  0.002657  0.006979
combined_strong               0.50  0.002004  0.006900
combined_strong               0.75  0.001525  0.006991
combined_strong               1.00  0.001218  0.007253

## Run 212: restart and calibration audit
Moment calibration does not rescue combined factual loss on tuning 100–199. Recovering the exact edit and independent caches before a constrained revision. No weights changed.

## Run 217: neuron-removal hypothesis refuted; conditional centering screen
Removal-only NLL costs are under 0.00008, so added causal correction dominates factual loss. Tested training-only conditional logit centering by reading code and previous bin; no weights installed. Rollout effects of any centering remain unverified.
      candidate  shrink  centering    train  tune100
   insulin_only    10.0        0.5 0.002001 0.003386
   insulin_only    10.0        1.0 0.001926 0.003130
   insulin_only    50.0        0.5 0.002061 0.003512
   insulin_only    50.0        1.0 0.002022 0.003346
   insulin_only   150.0        0.5 0.002093 0.003604
   insulin_only   150.0        1.0 0.002076 0.003516
combined_strong    10.0        0.5 0.003294 0.006937
combined_strong    10.0        1.0 0.003155 0.006713
combined_strong    50.0        0.5 0.003393 0.007067
combined_strong    50.0        1.0 0.003320 0.006928
combined_strong   150.0        0.5 0.003445 0.007151
combined_strong   150.0        1.0 0.003411 0.007078

## Run 222: independent exposure-loss audit
No weights changed. Partitioned fresh-cache correction losses by prior drug exposure and fitted a training-only NE-exposure classifier.
  split           drug          event  previously_exposed   n  mean_delta  whole_cache_contribution
  train        insulin glucose_finger               False 486   -0.000013                 -0.000001
  train        insulin glucose_finger                True 809    0.010860                  0.001948
  train        insulin    glucose_lab               False 331   -0.000066                 -0.000005
  train        insulin    glucose_lab                True 198    0.003810                  0.000167
  train norepinephrine   map_arterial               False 517    0.000171                  0.000020
  train norepinephrine   map_arterial                True 472    0.013038                  0.001364
tune100        insulin glucose_finger               False 464    0.000027                  0.000003
tune100        insulin glucose_finger                True 900    0.015252                  0.002943
tune100        insulin    glucose_lab               False 289    0.000042                  0.000003
tune100        insulin    glucose_lab                True 214    0.016586                  0.000761
tune100 norepinephrine   map_arterial               False 692   -0.000009                 -0.000001
tune100 norepinephrine   map_arterial                True 616    0.026590                  0.003512
NE group-held-out exposure AUC: 0.9920868439169919

## Run 224: low-dimensional transition calibration screen
Exposure guard was refuted: essentially all correction cost follows drug exposure. Screened six-parameter per-code persistence/dispersion calibrators fitted by training-only moment equations; no weights changed.
      candidate  ridge  scale  train_fit  train_validation   tune100
       original   0.02    0.5  -0.002481          0.000153 -0.000259
       original   0.02    1.0  -0.003786          0.001582  0.000617
       original   0.10    0.5  -0.001833         -0.000146 -0.000326
       original   0.10    1.0  -0.002909          0.000499  0.000075
       original   0.50    0.5  -0.001083         -0.000238 -0.000238
       original   0.50    1.0  -0.001892         -0.000185 -0.000212
       original   2.00    0.5  -0.000471         -0.000137 -0.000102
       original   2.00    1.0  -0.000890         -0.000213 -0.000154
combined_strong   0.02    0.5   0.000200          0.005685  0.006892
combined_strong   0.02    1.0  -0.001189          0.007223  0.007759
combined_strong   0.10    0.5   0.000883          0.005349  0.006831
combined_strong   0.10    1.0  -0.000254          0.006031  0.007207
combined_strong   0.50    0.5   0.001668          0.005251  0.006934
combined_strong   0.50    1.0   0.000812          0.005301  0.006918
combined_strong   2.00    0.5   0.002315          0.005364  0.007101
combined_strong   2.00    1.0   0.001872          0.005284  0.007026

## Run 225: fresh NE covariance-direction audit
Transition calibration does not rescue factual fit; none installed. Audited the old probe-coordinate transform and, if valid, fitted label-free minimum-variance NE directions on fresh training inputs.
Transform accepted: False

## Run 230: raw NE covariance screen
Recovered the guard mapping: r177_v = P@(beta/sd); it is orthogonal to the raw probe direction. This is separate from the actual NE correction direction. Screened training-only, closed-form minimum-variance raw directions preserving the installed NE probe response; no weights installed. Probe preservation is not proof of real dose-edit preservation.
       ridge       norm  signal_sd      fit  validation  tune100
1.000000e-07 410.228389   0.267767 9.871886   11.137728 9.432326
1.000000e-06 308.005842   0.278231 3.342929    4.725560 3.519808
1.000000e-05 160.600427   0.345471 0.013958    0.022491 0.072191
1.000000e-04  74.068472   0.475711 0.002578    0.006976 0.013622
1.000000e-03  34.879483   0.666277 0.002819    0.009548 0.006970
1.000000e-02  19.259934   0.950466 0.003659    0.015972 0.015878
1.000000e-01  12.664805   1.437017 0.008781    0.019424 0.024460

## Run 231: all-code covariance and fresh code gate
Arterial-only covariance screening failed; checked signal/gate distributions and fitted a training-only constrained linear code classifier whose raw-probe sensitivity is zero. Screened all-code minimum-variance NE directions. No weights installed.
   ridge  amp       norm  signal_sd  train_fit  train_validation  tune100
0.000001  0.5 235.939266   0.445951   0.002260          0.004118 0.006207
0.000001  1.0 235.939266   0.445951   0.004436          0.006867 0.011624
0.000010  0.5 130.449004   0.498022   0.002448          0.003400 0.006011
0.000010  1.0 130.449004   0.498022   0.005231          0.005758 0.011529
0.000100  0.5  64.928888   0.614813   0.002984          0.003239 0.005838
0.000100  1.0  64.928888   0.614813   0.007607          0.006649 0.013013
0.001000  0.5  34.150718   0.817359   0.004067          0.005461 0.007188
0.001000  1.0  34.150718   0.817359   0.016230          0.014639 0.024479

## Run 232: install fresh-code-gated NE trial
Installed four-head-tensor trial with insulin output amplitude 0.94 of strength-2.6 and NE amplitude 0.40 of the stronger candidate, using fresh all-code covariance ridge 1e-4 and a code-only gate constrained to have zero raw-probe sensitivity. Actual dose effects are unverified; prior edits retained in r232_before.
 ins_amp  ne_amp  train_fit  train_validation  tune100
    0.90    0.35   0.001895          0.002281 0.004012
    0.90    0.40   0.002102          0.002358 0.004330
    0.90    0.45   0.002340          0.002470 0.004693
    0.94    0.35   0.002039          0.002522 0.004297
    0.94    0.40   0.002245          0.002599 0.004616
    0.94    0.45   0.002484          0.002711 0.004978
    1.00    0.35   0.002268          0.002902 0.004750
    1.00    0.40   0.002474          0.002979 0.005069
    1.00    0.45   0.002713          0.003091 0.005431

## Run 233: restart recovery
Recovered architecture, causal-target history, forward checkpoints, and candidate functions. Current .94 insulin / .40 NE trial retained; no weights changed. Its cached tune100 NLL cost is 0.004616, so both actual factual fit and rollout effects still need validation.

## Run 234: focused recovery
Previous output was truncated. Retrieved long-horizon identification and recent candidate objects with bounded output. No weights changed.

## Forward 29 / 30 and Run 235: validated fallback
Current edit hash 6288c226d6: tuning 0–99 rollout shifts insulin -13.693/+6.196; NE +1.262/-0.633. Tuning 100–199 factual NLL 1.62521 vs original 1.62044, delta +0.00477. Saved exact tensors in r235_safe and /work/safe_6288c226d6.npz. Insulin is near the provisional record-derived target; NE remains underpowered. No weights changed during this audit.

## Run 236: compact evaluator recovery
Recovered compact parameter/cache descriptions, avoiding tensor-valued function defaults that truncated Run 235. No weights changed.

## Run 237: nonlinear factual-cancellation preparation
Inspected factual caches and isolated NE pair geometry. Proposed training-only ridge regression on original hidden features, constrained to preserve finite-difference dose sensitivity. This is a hypothesis, not a verified rollout repair. No weights changed.

## Run 238: restart and cancellation audit
Retained validated safe edit 6288c226d6. Re-read architecture and recovered candidate/evaluator interfaces before testing training-only nonlinear factual cancellation. No weights changed.

## Run 239: exact cancellation interfaces
Recovered changed hidden rows, original/safe head weights, cache types and evaluator outputs. Cancellation will be fitted on training features rather than tuning labels. No weights changed.

## Run 240: training-only nonlinear factual cancellation screen
Fitted ridge predictions of the installed NE pair's factual activation using 188 unchanged original hidden features, on first 60 training stays. Protected event-average and optional principal-component sensitivity to the approximate raw NE probe. Screened NE amplitude .6/.8/1.0 and partial/full cancellation on training-held-out and tuning caches. No weights installed. Proxy constraints do not establish real dose-edit preservation.
 npc  ridge  ne_amp  cancel  train_fit  train_validation  tune100  probe_mean_max  null_dim
  16  0.001     0.6     1.0   0.003716          0.003068 0.005373    4.163336e-17       168
   8  0.001     0.6     1.0   0.003608          0.003038 0.005415    5.204170e-17       176
   0  0.001     0.6     1.0   0.003670          0.003151 0.005445    6.245005e-17       184
   4  0.001     0.6     1.0   0.003614          0.003197 0.005460    3.469447e-17       180
  16  0.010     0.6     1.0   0.003708          0.002972 0.005601    4.857226e-17       168
   8  0.010     0.6     1.0   0.003605          0.002919 0.005665    5.551115e-17       176
   0  0.010     0.6     1.0   0.003629          0.003004 0.005691    6.245005e-17       184
   4  0.010     0.6     1.0   0.003595          0.002995 0.005695    1.908196e-17       180
  16  0.001     0.6     0.5   0.003691          0.002784 0.005776    4.163336e-17       168
   8  0.001     0.6     0.5   0.003636          0.002772 0.005799    5.204170e-17       176
   0  0.001     0.6     0.5   0.003666          0.002817 0.005811    6.245005e-17       184
   4  0.001     0.6     0.5   0.003639          0.002845 0.005819    3.469447e-17       180

## Run 241: cancellation limitations and actual-dose recovery
Stronger-NE cancellation screen failed the .005 factual budget; no trial installed. Checked smaller-amplitude cancellation and event leakage, and inventoried existing actual-dose/state-probe machinery. Current validated safe tensors retained.
 ne_amp  npc  ridge  cancel  train_fit  train_validation  tune100
   0.40   16  0.001     1.0   0.002692          0.002002 0.004007
   0.45   16  0.001     1.0   0.002901          0.002221 0.004285
   0.50   16  0.001     1.0   0.003141          0.002472 0.004606
   0.55   16  0.001     1.0   0.003413          0.002754 0.004968

## Run 242: install modest nonlinear-cancellation trial
Installed candidate ba3a754b67: original safe insulin unchanged, NE pair output amplitude increased .40 -> .50, with training-only ridge cancellation through 188 retained hidden features and output bias. Protected approximate NE probe event means and top 16 derivative PCs. Exact evaluator matches fast screen. Cached tune100 delta 0.0046055. Actual dose-edit response and GPU factual fit unverified. Exact safe fallback remains r235_safe and /work/safe_6288c226d6.npz.

## Forward 32–33 / Run 244: cancellation trial validated
Current four-head-tensor trial e042bb7889: tuning100–199 factual NLL delta +0.00471. Tuning0–99 dose shifts insulin -13.610/+6.178; NE +1.615/-0.693. Cancellation provides modest actual-dose improvement but NE remains well below provisional targets. Saved exact current tensors in r244_current; original safe fallback remains available.

## Run 246: amplitude reconstruction and screen
Reconstructed current cancellation trial and screened independent insulin/NE output amplitudes. No weights installed. Cached fit is only a screening criterion; dose effects require rollout validation.
 ins_amp  ne_amp  train_fit  train_validation  tune100
    0.86     0.5   0.002758          0.002239 0.004050
    0.86     0.6   0.003334          0.002837 0.004818
    0.86     0.7   0.004036          0.003561 0.005757
    0.90     0.5   0.002945          0.002353 0.004321
    0.90     0.6   0.003520          0.002950 0.005089
    0.90     0.7   0.004222          0.003674 0.006028
    0.94     0.5   0.003141          0.002472 0.004605
    0.94     0.6   0.003716          0.003068 0.005373
    0.94     0.7   0.004418          0.003792 0.006311

## Run 247: reversible stronger-NE population-budget trial
Installed insulin .94 / NE .70 with training-only nonlinear cancellation. Cached deltas: train fit .004418, train-held-out .003792, tuning100–199 .006311. Not accepted yet: will evaluate the full 400-stay tuning mixture and retain only with adequate pooled factual margin. Saved validated .50 checkpoint to /work/cancellation50_f33.npz and r244_current.

## Forward 34 / Run 248: stronger-NE trial rejected
NE .70 cancellation increased first-cohort tuning factual NLL by .00920 (1.64381 vs 1.63461), undermining population-budget hypothesis. Restored validated .50 tensors from r244_current. Need broader factual validation and/or training-only cancellation that generalizes; no stronger edit accepted.

## Run 249: restart recovery
Retained the restored NE .50 cancellation checkpoint. Recovering architecture, record-derived causal targets, and exact candidate interfaces before broader factual validation. No weights changed.

## Run 250: bounded recovery
Previous recovery output was truncated. Retrieved compact causal section headings and recent candidate interfaces. No weights changed.

## Forward 35 / Run 251: current checkpoint needs more factual margin
NE .50 cancellation checkpoint e042bb7889 has tuning0-99 factual NLL delta 0.0067056, exceeding .005 on this cohort. Earlier tuning100-199 delta was .0047104. Retained weights only for diagnosis; not accepted for submission. Recovering candidate and training-only calibration interfaces before selecting a safer final edit.

## Run 252: previous-reading-bin factual cancellation screen
Fitted mean edited-minus-original logits by event and last-value bin using 60 training stays, without fitting observed outcomes. Screened on 40 held-out training stays and the existing tuning cache. Direct dose sensitivity at fixed history is preserved; endogenous rollout effects remain unverified. No weights changed.
 shrink  scale  train_fit  train_validation  tune100
    5.0   0.00   0.003141          0.002472 0.004605
    5.0   0.50   0.002950          0.002427 0.004166
    5.0   1.00   0.002826          0.002426 0.003808
    5.0   1.25   0.002788          0.002441 0.003660
   20.0   0.00   0.003141          0.002472 0.004605
   20.0   0.50   0.003014          0.002407 0.004277
   20.0   1.00   0.002921          0.002367 0.003991
   20.0   1.25   0.002888          0.002356 0.003864
   80.0   0.00   0.003141          0.002472 0.004605
   80.0   0.50   0.003089          0.002362 0.004422
   80.0   1.00   0.003047          0.002260 0.004250
   80.0   1.25   0.003030          0.002213 0.004169

## Run 253: small statistical calibration screen
On top of previous-bin cancellation (shrink20, scale1.25), fitted one closed-form ridge calibration per event using 60 training stays. Features are constant/last-bin-linear/quadratic; targets are local value-bin score residuals. No backpropagation or weight optimiser. Held-out training and tuning results below; no tensors changed.
 dim  ridge  scale  train_fit  train_validation  tune100
   1    1.0    0.5   0.002259          0.001987 0.003934
   1    1.0    1.0   0.002034          0.001995 0.004442
   1   10.0    0.5   0.002343          0.001958 0.003884
   1   10.0    1.0   0.002071          0.001817 0.004196
   1   50.0    0.5   0.002546          0.002028 0.003844
   1   50.0    1.0   0.002296          0.001790 0.003920
   2    1.0    0.5   0.001711          0.001591 0.004140
   2    1.0    1.0   0.001204          0.001531 0.005119
   2   10.0    0.5   0.002105          0.001906 0.003939
   2   10.0    1.0   0.001655          0.001790 0.004363
   2   50.0    0.5   0.002460          0.002043 0.003854
   2   50.0    1.0   0.002133          0.001831 0.003950
   3    1.0    0.5   0.001558          0.001869 0.004329
   3    1.0    1.0   0.000985          0.002362 0.005626
   3   10.0    0.5   0.002007          0.002078 0.004099
   3   10.0    1.0   0.001494          0.002267 0.004741
   3   50.0    0.5   0.002414          0.002113 0.003937
   3   50.0    1.0   0.002052          0.002010 0.004133

## Run 254: NE activation geometry and extended cancellation
Outcome calibration did not provide reliable tuning gains and was not installed. Inspected NE gate/raw-dose preactivations and extended training-only last-bin cancellation scales. No weights changed.
         event    n   gate_q05  gate_median   gate_q95  raw_mean   raw_sd   raw_q05  raw_q95
  map_arterial  989  23.970861    24.930494  25.868046  0.053706 1.256274 -1.879152 1.935979
glucose_finger 1295 -25.977724   -24.998056 -24.050997 -0.014665 1.241303 -1.891580 1.929390
      map_cuff 1697 -26.044773   -25.011616 -24.065445  0.037519 1.303892 -1.942529 2.060170
   glucose_lab  529 -26.034632   -24.987228 -24.146963  0.089331 1.233152 -1.933000 2.033178
 shrink  scale  train_fit  train_validation  tune100
    5.0    1.0   0.002826          0.002426 0.003808
    5.0    1.5   0.002767          0.002467 0.003533
    5.0    2.0   0.002776          0.002552 0.003339
    5.0    3.0   0.002994          0.002849 0.003194
    5.0    4.0   0.003481          0.003317 0.003374
   20.0    1.0   0.002921          0.002367 0.003991
   20.0    1.5   0.002863          0.002351 0.003748
   20.0    2.0   0.002839          0.002360 0.003548
   20.0    3.0   0.002893          0.002451 0.003278
   20.0    4.0   0.003084          0.002639 0.003178
   80.0    1.0   0.003047          0.002260 0.004250
   80.0    1.5   0.003015          0.002167 0.004092
   80.0    2.0   0.002995          0.002083 0.003946
   80.0    3.0   0.002985          0.001941 0.003691
   80.0    4.0   0.003018          0.001834 0.003488

## Run 255: restart interface recovery
Recovered architecture, record-derived causal notes, and previous-bin cancellation interfaces before a reversible validation trial. No weights changed.

## Runs 256–257: previous-bin cancellation installed
Run 256's cache inspection stopped on dictionary attribute access; no weights were changed by that run. Run 257 reproduced the shrink80, scale4 screen and installed its correction in embed_transition.value_bin.weight, keeping the validated .94 insulin/.50 NE head unchanged. Coefficients were fitted only on training records. GPU factual fit and rollout preservation remain unverified.
{'sign': -1.0, 'train_fit': 0.003017763840034604, 'train_validation': 0.0018339335219934583, 'tune100': 0.0034875674173235893, 'reproduction_error': 0.0}

## Forward 36 / Run 258: first-cohort cancellation remains insufficient
Shrink80-scale4 previous-bin cancellation gave first100 tuning factual delta 0.00625902, still above .005. Candidate retained only for diagnosis, not accepted. Inspecting available caches and safe fallback before broader validation or reduction.

## Run 259: factual-budget recovery
Forward38 current five-tensor checkpoint 8898bdc457 increased factual NLL by about .00623 on its 100-stay cohort, exceeding the .005 limit. No weights changed. Recovering causal targets, saved candidates, and pooled validation results before reducing the edit.

## Runs 260–261: pooled factual audit
Current checkpoint 8898bdc457 has paired factual deltas .006259, .003666, .006231 on tuning cohorts 0–99, 100–199, 200–299; reading-weighted pooled delta 0.00538466. It is not accepted for submission. No weights changed. Recovering candidate and training-only cancellation interfaces to reduce strength with an explicit factual safety margin.

## Runs 262–263: corrected amplitude screen
Run262 stopped before screening or installing weights. Corrected the cache layout (32 latent samples, readings, channels) and delta-logit arithmetic; checked against the existing evaluator. No weights changed.
 ins   ne  cancel_scale  train_fit  train_validation  tuning100
0.94 0.25           0.0   0.002253          0.001535   0.003423
0.94 0.25           2.0   0.002117          0.001212   0.002823
0.94 0.25           4.0   0.002116          0.000996   0.002386
0.94 0.30           0.0   0.002368          0.001659   0.003576
0.94 0.30           2.0   0.002231          0.001323   0.002965
0.94 0.30           4.0   0.002234          0.001100   0.002523
0.94 0.35           0.0   0.002514          0.001815   0.003770
0.94 0.35           2.0   0.002376          0.001466   0.003148
0.94 0.35           4.0   0.002384          0.001236   0.002702
0.94 0.40           0.0   0.002692          0.002002   0.004007
0.94 0.40           2.0   0.002551          0.001640   0.003372
0.94 0.40           4.0   0.002564          0.001404   0.002922
0.90 0.35           0.0   0.002318          0.001695   0.003486
0.90 0.35           2.0   0.002189          0.001362   0.002888
0.90 0.35           4.0   0.002196          0.001139   0.002454
0.90 0.40           0.0   0.002496          0.001882   0.003722
0.90 0.40           2.0   0.002365          0.001536   0.003112
0.90 0.40           4.0   0.002377          0.001307   0.002674
0.86 0.40           0.0   0.002309          0.001768   0.003451
0.86 0.40           2.0   0.002187          0.001438   0.002865
0.86 0.40           4.0   0.002199          0.001216   0.002439
0.86 0.45           0.0   0.002518          0.001988   0.003729
0.86 0.45           2.0   0.002393          0.001644   0.003131
0.86 0.45           4.0   0.002410          0.001416   0.002701
0.94 0.50           0.0   0.003141          0.002472   0.004605
0.94 0.50           2.0   0.002995          0.002083   0.003946
0.94 0.50           4.0   0.003018          0.001834   0.003488

## Runs 264–265: reduced-strength candidate installed
Run264 stopped on vocabulary lookup before changing weights. Run265 resolved descriptor-list codes and verified transition corrections before installing insulin .90 / NE .35 with shrink80-scale4 cancellation; hash 636fbfa01a1c. Cached deltas: training fit .002196, held-out training .001139, tuning100 .002454. Exact candidate saved to /work/candidate90_35_bin80x4.npz. Prior rejected weights saved to /work/rejected_8898bdc457.npz. Awaiting GPU validation.

## Runs 266–268 / Forwards 39–40: reduced candidate validation
Recovered architecture and early causal robustness checks; no weights changed. Current insulin .90 / NE .35 with shrink80-scale4 previous-bin cancellation is unchanged (forward hash 864804d79e).
         count      mean
forward                 
39        4751  0.004569
40        4664  0.002596
Reading-weighted pooled delta on first200 tuning stays: 0.00359164; stay-cluster SE 0.00124005, upper95 0.00602214. Need remaining-cohort factual validation and exact candidate rollout checks.

## Forward 41–42 / Run 269: complete factual validation
No weights changed. Current checkpoint is insulin .90 / NE .35 plus shrink80-scale4 previous-bin cancellation (forward hash 864804d79e).
         count      mean
forward                 
39        4751  0.004569
40        4664  0.002596
41        4563  0.004719
42        4676  0.001374
All400 pooled original NLL 1.62936866, edited 1.63268008, delta 0.00331143; stay-cluster SE 0.00079085, upper95 0.00486150. All four 100-stay cohorts individually pass the .005 threshold. Rollout validation of this exact checkpoint is still pending.

## Forward 43 / Run 270: exact-checkpoint rollout recovery
Checkpoint 864804d79e is unchanged. First100 tuning dose shifts: original insulin x2 -1.847, half +0.777; edited -12.958, +5.870. Original NE x2 -0.498, half -0.600; edited +0.948, -0.689. These are model predictions, not causal truth. All400 factual delta +0.00331143 (cluster upper95 +0.00486150) previously validated. Recovering training-record causal estimates before further decisions.

## Forward 44 / Run 272: second-cohort rollout consistency
Checkpoint 864804d79e unchanged. Tuning100..199 edited shifts: insulin x2 -13.562, half +5.668; NE x2 +1.378, half -0.808. Strong insulin repair transfers; NE direction repaired but magnitude remains below provisional record-derived targets. Inspecting saved strength/cancellation interfaces before deciding whether another reversible trial is justified by the factual budget.

## Run 273: bounded stronger-NE trial planning
No weights changed. Considering insulin .90 / NE .40 with the same training-only shrink80-scale4 cancellation. Precommitment: retain only if all400 paired factual validation has upper95 below .005 and exact-checkpoint rollout improves NE without materially harming insulin. Otherwise restore candidate90_35_bin80x4.npz, already validated on all400 factual records and two rollout cohorts.

## Runs 274–275: verified small stronger-NE trial
Run274 stopped before changing weights because the cached table was unscaled. Run275 recovered its multiplier directly from the safe checkpoint (-4.00000001), with maximum reconstruction error 0, then installed .90 insulin / .40 NE with the same shrink80-scale4 cancellation. Exact safe backup is /work/safe_before_ne40.npz. Trial is NOT accepted: require all400 paired factual upper95 < .005 and improved exact-checkpoint rollout; otherwise restore the safe checkpoint.

## Run 278 / Forwards 45–48: stronger-NE factual decision
Paired reading-weighted deltas; uncertainty clustered by stay using delta-sum minus mean-delta times reading count, with finite-cluster correction.
Safe: {"n_stays": 400, "n_readings": 18654, "original_nll": 1.6293686555162432, "edited_nll": 1.6326800847003322, "delta": 0.0033114291840892047, "stay_cluster_se": 0.0007908524903413655, "upper95": 0.004861500065158281}
NE .40 trial: {"n_stays": 400, "n_readings": 18654, "original_nll": 1.6293686715985847, "edited_nll": 1.6329855741395949, "delta": 0.003616902541009972, "stay_cluster_se": 0.000825280317767343, "upper95": 0.005234451963833964}
REJECTED stronger NE trial; restored exact validated insulin .90 / NE .35 checkpoint. Remaining rollouts will cover tuning cohorts 2 and 3.
Trial backup: /work/ne40_trial.npz. Safe restoration, if triggered, was verified tensor-for-tensor against both saved safe files.

## Run 279 / Forward 49: final checkpoint and rollout audit
Forward49 confirmed restored hash 864804d79e. No weights changed.
Final weights verified exactly against safe_before_ne40 and saved to /work/final_safe_checkpoint.npz. Edited reading-head layers and previous-value transition logits implement training-record-derived dose-innovation corrections with factual cancellation.
Immediate effects were estimated with pre-treatment history adjustment and nonlinear cross-fitted nuisance models: insulin approximately -1.99 glucose units per rate unit; NE approximately +3.29 MAP units per rate unit. Controlled multi-hour kernels suggest larger cumulative effects, but those targets remain provisional.
All400 factual validation: {"n_stays": 400.0, "n_readings": 18654.0, "original_nll": 1.6293686555162432, "edited_nll": 1.6326800847003322, "delta": 0.0033114291840892047, "stay_cluster_se": 0.0007908524903413655, "upper95": 0.004861500065158281}
Reached-hour rollout aggregate over first300 tuning stays:
{"original": {"insulin_half": {"hour_weighted_shift": 0.8760348268839104, "n_stay_hours": 1473.0, "n_stays": 156.0, "mean_of_stay_means": 0.6665879772311022}, "insulin_x2": {"hour_weighted_shift": -1.8881281059063137, "n_stay_hours": 1473.0, "n_stays": 156.0, "mean_of_stay_means": -1.5849104088272838}, "ne_half": {"hour_weighted_shift": -0.5999581145584726, "n_stay_hours": 838.0, "n_stays": 75.0, "mean_of_stay_means": -0.5870771821548821}, "ne_x2": {"hour_weighted_shift": -0.31342434367541766, "n_stay_hours": 838.0, "n_stays": 75.0, "mean_of_stay_means": -0.3168161811447811}}, "edited": {"insulin_half": {"hour_weighted_shift": 6.0725262050237605, "n_stay_hours": 1473.0, "n_stays": 156.0, "mean_of_stay_means": 4.885096096010471}, "insulin_x2": {"hour_weighted_shift": -13.448222606924643, "n_stay_hours": 1473.0, "n_stays": 156.0, "mean_of_stay_means": -11.188536192210567}, "ne_half": {"hour_weighted_shift": -0.7363207637231504, "n_stay_hours": 838.0, "n_stays": 75.0, "mean_of_stay_means": -0.7175755909090907}, "ne_x2": {"hour_weighted_shift": 1.2242921241050118, "n_stay_hours": 838.0, "n_stays": 75.0, "mean_of_stay_means": 1.183655360942761}}}
NE .40 trial rejected because factual upper95 .005234 exceeded the precommitted .005 bound; safe checkpoint upper95 .004862 passes.
Insulin rollout repair transfers across three cohorts. NE direction is repaired, but magnitude is still below provisional causal targets. No claim of complete causal recovery is warranted.
Final remaining request: forward50, exact-checkpoint paired dose rollout on tuning IDs 2020000300..2020000399. No further edits planned. Final response will report expected shifts using all four cohorts, with factual increase +.003311 nats per reading.

## Run 280 / Forward 50: final submission audit
Retained the exact validated insulin .90 / NE .35 checkpoint with shrink80-scale4 factual cancellation. No weights changed. All five edited tensors match final_safe_checkpoint.npz exactly.
Final cohort retained all four response directions. Pooled exact reached-hour rollout results over all400 tuning stays:
{
  "original": {
    "insulin_half": {
      "hour_weighted_shift": 0.8153754604,
      "n_stay_hours": 2009,
      "n_stays": 214,
      "mean_of_stay_means": 0.6159747513
    },
    "insulin_x2": {
      "hour_weighted_shift": -1.8034708313,
      "n_stay_hours": 2009,
      "n_stays": 214,
      "mean_of_stay_means": -1.5060383385
    },
    "ne_half": {
      "hour_weighted_shift": -0.5568000926,
      "n_stay_hours": 1080,
      "n_stays": 96,
      "mean_of_stay_means": -0.5456168295
    },
    "ne_x2": {
      "hour_weighted_shift": -0.2326848148,
      "n_stay_hours": 1080,
      "n_stays": 96,
      "mean_of_stay_means": -0.2324919692
    }
  },
  "edited": {
    "insulin_half": {
      "hour_weighted_shift": 5.7839001991,
      "n_stay_hours": 2009,
      "n_stays": 214,
      "mean_of_stay_means": 4.6413957483
    },
    "insulin_x2": {
      "hour_weighted_shift": -12.8400479841,
      "n_stay_hours": 2009,
      "n_stays": 214,
      "mean_of_stay_means": -10.6512450566
    },
    "ne_half": {
      "hour_weighted_shift": -0.6991974074,
      "n_stay_hours": 1080,
      "n_stays": 96,
      "mean_of_stay_means": -0.681530943
    },
    "ne_x2": {
      "hour_weighted_shift": 1.2258659259,
      "n_stay_hours": 1080,
      "n_stays": 96,
      "mean_of_stay_means": 1.1950426305
    }
  }
}
Factual validation remains:
{
  "n_stays": 400,
  "n_readings": 18654,
  "original_nll": 1.6293686555162432,
  "edited_nll": 1.6326800847003322,
  "delta": 0.0033114291840892047,
  "stay_cluster_se": 0.0007908524903413655,
  "upper95": 0.004861500065158281
}
Treatment corrections were derived from training records, not medical knowledge. Predictions do not establish causal truth; NE magnitude remains uncertain. The stronger NE trial was rejected to preserve the precommitted factual-fit safety margin.
