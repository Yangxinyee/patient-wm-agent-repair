# World-model weight repair

## Constraints
- Infer drug effects solely from the supplied simulated records.
- No model gradients or weight optimiser.
- Preserve factual value NLL within +0.005 nats of original.
- Maximum 300 Python runs and 50 forward requests.

## Initial plan
Read architecture and schema; examine treatment assignment and temporal variation; estimate effects with falsification checks; construct a targeted weight edit; validate factual fit and the exact rollout.

## Run 1
Inspecting architecture, tensor shapes, vocabulary, and sample records. No effect estimates or edits yet.

## Run 2
Architecture has separate observation and action encoders, with actions entering the latent posterior and recurrent prelude. Attribute heads also use last-value registers. This offers targeted editing routes, but no drug-effect conclusion has been drawn. Loaded both record files and inspected schema, vocabulary, distributions, and event timing.

## Run 3
Constructed complete hourly panels (-1 through 24), aligning infusions at hour+0.0833 to that hour's pre-dose measurements. Missing infusion records mean zero exposure. Auditing paired measurement channels, dose-assignment rules, and next-hour associations before estimating causal effects. No edits yet.

## Original rollout, forward 1
100 tuning stays: insulin x2 -3.226; insulin half +1.825; NE x2 +1.373; NE half -0.846. These are predictions, not truth.

## Run 4: dose-threshold identification
{
  "thresholds": [
    151.0,
    181.0,
    220.0,
    260.0
  ],
  "RD": [
    {
      "outcome": "next_finger",
      "bandwidth": 5,
      "adjusted": true,
      "n": 81498,
      "dose_jump": 0.5672,
      "effect_per_unit": -2.5211,
      "cluster_se": 0.3321,
      "ci95": [
        -3.172,
        -1.8702
      ]
    },
    {
      "outcome": "next_glucose_combined",
      "bandwidth": 5,
      "adjusted": true,
      "n": 83728,
      "dose_jump": 0.5985,
      "effect_per_unit": -2.4588,
      "cluster_se": 0.2953,
      "ci95": [
        -3.0375,
        -1.8801
      ]
    },
    {
      "outcome": "past_glucose_PLACEBO",
      "bandwidth": 5,
      "adjusted": true,
      "n": 84494,
      "dose_jump": 0.6753,
      "effect_per_unit": 0.0984,
      "cluster_se": 0.0776,
      "ci95": [
        -0.0537,
        0.2505
      ]
    },
    {
      "outcome": "next_finger_observed",
      "bandwidth": 5,
      "adjusted": true,
      "n": 92540,
      "dose_jump": 0.6791,
      "effect_per_unit": 0.2774,
      "cluster_se": 0.0054,
      "ci95": [
        0.2667,
        0.2881
      ]
    },
    {
      "outcome": "next_finger",
      "bandwidth": 10,
      "adjusted": true,
      "n": 162730,
      "dose_jump": 0.7679,
      "effect_per_unit": -1.9114,
      "cluster_se": 0.1745,
      "ci95": [
        -2.2535,
        -1.5694
      ]
    },
    {
      "outcome": "next_glucose_combined",
      "bandwidth": 10,
      "adjusted": true,
      "n": 167103,
      "dose_jump": 0.7909,
      "effect_per_unit": -1.9249,
      "cluster_se": 0.1574,
      "ci95": [
        -2.2333,
        -1.6165
      ]
    },
    {
      "outcome": "past_glucose_PLACEBO",
      "bandwidth": 10,
      "adjusted": true,
      "n": 168455,
      "dose_jump": 0.8416,
      "effect_per_unit": 0.0146,
      "cluster_se": 0.0449,
      "ci95": [
        -0.0735,
        0.1027
      ]
    },
    {
      "outcome": "next_finger_observed",
      "bandwidth": 10,
      "adjusted": true,
      "n": 184729,
      "dose_jump": 0.8458,
      "effect_per_unit": 0.2728,
      "cluster_se": 0.0035,
      "ci95": [
        0.266,
        0.2796
      ]
    },
    {
      "outcome": "next_finger",
      "bandwidth": 14,
      "adjusted": true,
      "n": 227614,
      "dose_jump": 0.836,
      "effect_per_unit": -1.9368,
      "cluster_se": 0.1364,
      "ci95": [
        -2.2041,
        -1.6694
      ]
    },
    {
      "outcome": "next_glucose_combined",
      "bandwidth": 14,
      "adjusted": true,
      "n": 233566,
      "dose_jump": 0.8548,
      "effect_per_unit": -1.9341,
      "cluster_se": 0.1233,
      "ci95": [
        -2.1757,
        -1.6925
      ]
    },
    {
      "outcome": "past_glucose_PLACEBO",
      "bandwidth": 14,
      "adjusted": true,
      "n": 234179,
      "dose_jump": 0.8938,
      "effect_per_unit": -0.0501,
      "cluster_se": 0.036,
      "ci95": [
        -0.1206,
        0.0204
      ]
    },
    {
      "outcome": "next_finger_observed",
      "bandwidth": 14,
      "adjusted": true,
      "n": 257262,
      "dose_jump": 0.8981,
      "effect_per_unit": 0.271,
      "cluster_se": 0.0029,
      "ci95": [
        0.2652,
        0.2768
      ]
    },
    {
      "outcome": "next_glucose_combined",
      "bandwidth": 5,
      "adjusted": false,
      "n": 83728,
      "dose_jump": 0.5986,
      "effect_per_unit": -2.6227,
      "cluster_se": 0.3245,
      "ci95": [
        -3.2587,
        -1.9867
      ]
    },
    {
      "outcome": "next_MAP_cross_drug_PLACEBO",
      "bandwidth": 10,
      "adjusted": true,
      "n": 172018,
      "dose_jump": 0.8467,
      "effect_per_unit": 0.0128,
      "cluster_se": 0.0793,
      "ci95": [
        -0.1426,
        0.1681
      ]
    }
  ]
}

## Run 5: recording-policy falsification
Initial insulin RD was accompanied by a large discontinuity in future finger-reading availability, threatening identification. Rounded threshold values have half-jumps, so this audit uses cuts at 150/180/220/260 and excludes exact cut values. Testing lab outcomes and their availability separately, and unadjusted past-outcome placebos rather than treating history-adjusted placebos as decisive.
[
  {
    "outcome": "next_LAB",
    "bandwidth": 5,
    "adjusted": true,
    "n": 18414,
    "dose_jump": 1.033,
    "effect_per_unit": -1.5496,
    "cluster_se": 0.3235,
    "ci95": [
      -2.1836,
      -0.9155
    ]
  },
  {
    "outcome": "next_FINGER",
    "bandwidth": 5,
    "adjusted": true,
    "n": 80147,
    "dose_jump": 1.0508,
    "effect_per_unit": -1.8271,
    "cluster_se": 0.2196,
    "ci95": [
      -2.2576,
      -1.3967
    ]
  },
  {
    "outcome": "next_LAB",
    "bandwidth": 10,
    "adjusted": true,
    "n": 36529,
    "dose_jump": 1.0337,
    "effect_per_unit": -1.5257,
    "cluster_se": 0.2116,
    "ci95": [
      -1.9405,
      -1.1108
    ]
  },
  {
    "outcome": "next_FINGER",
    "bandwidth": 10,
    "adjusted": true,
    "n": 161183,
    "dose_jump": 1.0415,
    "effect_per_unit": -1.7381,
    "cluster_se": 0.1439,
    "ci95": [
      -2.0202,
      -1.456
    ]
  },
  {
    "outcome": "next_LAB",
    "bandwidth": 14,
    "adjusted": true,
    "n": 51089,
    "dose_jump": 1.0281,
    "effect_per_unit": -1.5708,
    "cluster_se": 0.1784,
    "ci95": [
      -1.9205,
      -1.2211
    ]
  },
  {
    "outcome": "next_FINGER",
    "bandwidth": 14,
    "adjusted": true,
    "n": 225930,
    "dose_jump": 1.0422,
    "effect_per_unit": -1.8228,
    "cluster_se": 0.1194,
    "ci95": [
      -2.0569,
      -1.5888
    ]
  },
  {
    "outcome": "next_LAB_observed",
    "bandwidth": 10,
    "adjusted": true,
    "n": 183867,
    "dose_jump": 1.043,
    "effect_per_unit": 0.0017,
    "cluster_se": 0.0039,
    "ci95": [
      -0.0059,
      0.0092
    ]
  },
  {
    "outcome": "next_FINGER_observed",
    "bandwidth": 10,
    "adjusted": true,
    "n": 183867,
    "dose_jump": 1.043,
    "effect_per_unit": 0.2702,
    "cluster_se": 0.0032,
    "ci95": [
      0.2639,
      0.2765
    ]
  },
  {
    "outcome": "next_CUFF_cross_drug",
    "bandwidth": 10,
    "adjusted": true,
    "n": 112798,
    "dose_jump": 1.0442,
    "effect_per_unit": 0.0713,
    "cluster_se": 0.0959,
    "ci95": [
      -0.1168,
      0.2594
    ]
  },
  {
    "outcome": "next_CUFF_observed",
    "bandwidth": 10,
    "adjusted": true,
    "n": 183867,
    "dose_jump": 1.043,
    "effect_per_unit": 0.0005,
    "cluster_se": 0.0038,
    "ci95": [
      -0.0069,
      0.0078
    ]
  },
  {
    "outcome": "past_LAB_unadjusted_PLACEBO",
    "bandwidth": 10,
    "adjusted": false,
    "n": 39744,
    "dose_jump": 1.0326,
    "effect_per_unit": -0.0321,
    "cluster_se": 0.2233,
    "ci95": [
      -0.4697,
      0.4056
    ]
  },
  {
    "outcome": "past_GLUCOSE_unadjusted_PLACEBO",
    "bandwidth": 10,
    "adjusted": false,
    "n": 167164,
    "dose_jump": 1.0431,
    "effect_per_unit": -0.1191,
    "cluster_se": 0.1328,
    "ci95": [
      -0.3793,
      0.1412
    ]
  }
]

## Run 6: independent effect checks
Clean threshold RD supports insulin next-hour effect about -1.55/unit on lab readings. Lab availability is smooth at thresholds; unadjusted past-outcome placebos are null. Finger availability jumps, and its effect estimate is more negative. Fitting stay-crossfitted history-adjusted effects and two-sensor transition models.
{
  "DML": [
    {
      "label": "insulin_next_lab",
      "drug": "insulin",
      "n": 80365,
      "effect_per_rate_unit": -1.9687,
      "cluster_se": 0.0734,
      "ci95": [
        -2.1127,
        -1.8248
      ],
      "dose_residual_sd": 0.4903,
      "outcome_residual_sd": 8.957,
      "seconds": 1.0
    },
    {
      "label": "NE_next_cuff_both_current_MAP",
      "drug": "norepinephrine",
      "n": 29293,
      "effect_per_rate_unit": 3.4255,
      "cluster_se": 0.442,
      "ci95": [
        2.5591,
        4.2918
      ],
      "dose_residual_sd": 0.0935,
      "outcome_residual_sd": 6.7848,
      "seconds": 0.7
    },
    {
      "label": "NE_next_cuff_any_current_MAP",
      "drug": "norepinephrine",
      "n": 240000,
      "effect_per_rate_unit": 2.7802,
      "cluster_se": 0.2277,
      "ci95": [
        2.334,
        3.2265
      ],
      "dose_residual_sd": 0.0659,
      "outcome_residual_sd": 7.0783,
      "seconds": 1.6
    }
  ],
  "transitions": [
    {
      "drug": "insulin",
      "n": 15905,
      "sensor_coefficients": [
        0.1525,
        0.8854
      ],
      "sum_sensor_coefficients": 1.0378,
      "drug_coefficient": -1.9235,
      "cluster_se": 0.1052,
      "intercept": -4.6136
    },
    {
      "drug": "norepinephrine",
      "n": 29293,
      "sensor_coefficients": [
        0.7326,
        0.19
      ],
      "sum_sensor_coefficients": 0.9226,
      "drug_coefficient": -2.0861,
      "cluster_se": 0.1752,
      "intercept": 8.5973
    }
  ]
}

## Run 10: NE threshold audit (mask fixed)
      sensor                outcome  bw  adjusted      n  dose_jump  outcome_jump    effect  cluster_se
map_arterial          map_cuff_next   4      True  14427    0.03949       0.10173   2.57572     7.34530
map_arterial      map_arterial_next   4      True  45668    0.04012       0.16659   4.15265     2.69603
map_arterial          map_cuff_next   8      True  29723    0.03873       0.08150   2.10442     4.77478
map_arterial      map_arterial_next   8      True  93919    0.03999      -0.02023  -0.50596     1.75248
map_arterial          map_cuff_next  12      True  45540    0.03507      -0.30378  -8.66193     4.31829
map_arterial      map_arterial_next  12      True 144456    0.03574      -0.17713  -4.95646     1.66000
map_arterial     map_cuff_next_seen   8      True  98906    0.04069       0.00072   0.01771     0.16178
map_arterial map_arterial_next_seen   8      True  98906    0.04069       0.00101   0.02484     0.07760
map_arterial      map_arterial_lag1   8     False  93961    0.05156      -0.17228  -3.34155     1.44330
    map_cuff          map_cuff_next   4      True  27302    0.03327      -0.21671  -6.51313     7.29831
    map_cuff      map_arterial_next   4      True  13039    0.00387      -0.21063 -54.42049   101.39637
    map_cuff          map_cuff_next   8      True  59367    0.03996      -0.07977  -1.99633     3.87124
    map_cuff      map_arterial_next   8      True  26457    0.00512      -0.11754 -22.97372    31.59373
    map_cuff          map_cuff_next  12      True  99803    0.03630      -0.49783 -13.71511     3.50214
    map_cuff      map_arterial_next  12      True  40266    0.00982      -0.22540 -22.96416    13.02311
    map_cuff     map_cuff_next_seen   8      True  84545    0.03056       0.00536   0.17539     0.18920
    map_cuff map_arterial_next_seen   8      True  84545    0.03056       0.00070   0.02297     0.08093
    map_cuff          map_cuff_lag1   8     False  59229    0.04852      -0.43193  -8.90167     3.23389
State sample IDs: [1010026818, 1010012893, 1010007807, 1010007891, 1010020515, 1010022688, 1010005770, 1010001192, 1010014800, 1010029265, 1010021227, 1010001869, 1010009418, 1010012484, 1010025432, 1010000431, 1010025790, 1010012667, 1010001013, 1010021897, 1010022639, 1010005551, 1010023750, 1010023586, 1010024158, 1010029313, 1010012776, 1010000682, 1010022805, 1010020817, 1010020491, 1010023984, 1010009687, 1010001133, 1010007386, 1010012456, 1010004932, 1010014466, 1010012099, 1010022844, 1010018371, 1010013764, 1010024238, 1010013214, 1010005806, 1010022808, 1010024920, 1010004574, 1010007269, 1010028979, 1010003888, 1010021234, 1010011733, 1010009334, 1010018070, 1010021339, 1010019214, 1010024615, 1010012666, 1010022540, 1010008744, 1010019327, 1010012582, 1010020287, 1010016071, 1010015462, 1010012535, 1010025327, 1010026671, 1010011223, 1010006811, 1010029094, 1010009439, 1010022564, 1010006079, 1010006618, 1010022570, 1010024912, 1010018574, 1010007662, 1010005808, 1010018322, 1010026881, 1010008553, 1010018559, 1010019209, 1010026994, 1010000842, 1010011240, 1010001344, 1010024931, 1010029456, 1010011674, 1010009179, 1010011705, 1010026618, 1010020196, 1010003937, 1010025179, 1010021143]

## Run 11: rich pre-dose history, stay-crossfitted estimates
                    label      n   effect  cluster_se  dose_residual_sd  outcome_residual_sd  seconds
     NE_cuff_both_current  28031  3.51113     0.46432           0.09161              6.82217      1.5
 NE_arterial_both_current  89388  3.91356     0.16693           0.09039              4.27770      2.1
    NE_cuff_all_histories 250000  3.61924     0.23487           0.06341              7.05825      3.6
insulin_lab_all_histories 137825 -1.96674     0.07393           0.37319              9.97434      2.5
States from 100 randomly sampled training stays loaded as S; history features F.

## Run 12: persistence and rollout target sensitivity
          drug     n  assumed_effect  persistence  cluster_se  cross_sensor_residual_cov
       insulin 15266           -1.55      0.99773     0.00781                  108.17329
       insulin 15266           -1.80      1.00763     0.00782                  108.17329
       insulin 15266           -2.00      1.01556     0.00784                  108.17329
norepinephrine 28031            3.00      0.89761     0.02096                   10.64188
norepinephrine 28031            3.60      0.89803     0.02098                   10.64188
norepinephrine 28031            4.00      0.89831     0.02099                   10.64188
          drug     b    a  eligible_stays       x2    half
       insulin -1.55 0.80             205  -6.8094  3.4047
       insulin -1.55 0.90             205  -9.0062  4.5031
       insulin -1.55 0.95             205 -10.5409  5.2705
       insulin -1.80 0.80             205  -7.9077  3.9539
       insulin -1.80 0.90             205 -10.4588  5.2294
       insulin -1.80 0.95             205 -12.2411  6.1205
       insulin -2.00 0.80             205  -8.7864  4.3932
       insulin -2.00 0.90             205 -11.6209  5.8105
       insulin -2.00 0.95             205 -13.6012  6.8006
norepinephrine  3.00 0.70             155   3.8398 -1.9199
norepinephrine  3.00 0.80             155   4.9006 -2.4503
norepinephrine  3.00 0.90             155   6.5400 -3.2700
norepinephrine  3.60 0.70             155   4.6077 -2.3039
norepinephrine  3.60 0.80             155   5.8808 -2.9404
norepinephrine  3.60 0.90             155   7.8479 -3.9240
norepinephrine  4.00 0.70             155   5.1197 -2.5598
norepinephrine  4.00 0.80             155   6.5342 -3.2671
norepinephrine  4.00 0.90             155   8.7199 -4.3600
Targets assume linear, time-homogeneous drug effects; they are estimates, not tool-provided truth.

## Restart, run 13: recovering architecture and kernel state
No edits yet. Drug-effect evidence is promising, but glucose persistence near one requires further checks before fixing rollout targets. Inspecting architecture to identify a constrained, factual-preserving edit.

## Run 16: local value-head reconstruction
Forward 4 provides the factual baseline on the 100 state-sampled training stays: NLL 1.64320, 4660 readings. Testing activation, query type, gap scaling, and last-register layout before relying on a head-only edit.
[{"activation":"gelu","gap_mode":"seconds_div10","type_id":1,"status_id":0,"age_mode":"seconds_div10","zcol":2,"acol":1,"flags":[0,3],"mean_rmse":3.5798306793,"nll_rmse":0.4059037833,"score":4.7975420293},{"activation":"gelu","gap_mode":"seconds_div10","type_id":1,"status_id":0,"age_mode":"hours_div10","zcol":2,"acol":1,"flags":[0,3],"mean_rmse":3.723884087,"nll_rmse":0.3868159534,"score":4.8843319473},{"activation":"silu","gap_mode":"seconds_div10","type_id":1,"status_id":0,"age_mode":"hours_div10","zcol":2,"acol":1,"flags":[0,3],"mean_rmse":4.1737970911,"nll_rmse":0.4578360787,"score":5.5473053274},{"activation":"relu","gap_mode":"seconds_div10","type_id":1,"status_id":0,"age_mode":"seconds_div10","zcol":2,"acol":1,"flags":[0,3],"mean_rmse":4.4656973242,"nll_rmse":0.4518974922,"score":5.8213898009},{"activation":"relu","gap_mode":"seconds_div10","type_id":1,"status_id":0,"age_mode":"hours_div10","zcol":2,"acol":1,"flags":[0,3],"mean_rmse":4.9303510315,"nll_rmse":0.4887269886,"score":6.3965319972}]

## Run 17: head reconstruction audit
[{"gap_activation":"silu","prior_activation":"gelu","value_activation":"silu","bin_side":"left","zmode":"prior_mc","mean_rmse":3.3804607007,"nll_rmse":0.4042454013,"mean_bias":0.7131807546,"score":4.5931969045},{"gap_activation":"gelu","prior_activation":"gelu","value_activation":"silu","bin_side":"left","zmode":"prior_mean","mean_rmse":3.4121065303,"nll_rmse":0.4161632608,"mean_bias":0.2942680101,"score":4.6605963127},{"gap_activation":"gelu","prior_activation":"gelu","value_activation":"gelu","bin_side":"left","zmode":"prior_mc","mean_rmse":3.4678569554,"nll_rmse":0.4046162167,"mean_bias":0.1949307279,"score":4.6817056056},{"gap_activation":"gelu","prior_activation":"gelu","value_activation":"silu","bin_side":"left","zmode":"prior_mc","mean_rmse":3.4314455566,"nll_rmse":0.4190449924,"mean_bias":0.159751666,"score":4.6885805339},{"gap_activation":"silu","prior_activation":"gelu","value_activation":"silu","bin_side":"left","zmode":"prior_mean","mean_rmse":3.4818419202,"nll_rmse":0.4023464681,"mean_bias":0.8598995141,"score":4.6888813245}]
[{"event":"map_cuff","n":141,"mean_rmse":2.4984212972,"mean_bias":-0.2137319476,"nll_rmse":0.4132155496,"nll_bias":0.0744951858,"predicted_four_draw_mean_rmse":1.2742730379},{"event":"map_arterial","n":92,"mean_rmse":2.0309635603,"mean_bias":-0.1182088646,"nll_rmse":0.4506170404,"nll_bias":0.0312715331,"predicted_four_draw_mean_rmse":0.9304040074},{"event":"glucose_finger","n":121,"mean_rmse":4.9379716184,"mean_bias":2.9124298134,"nll_rmse":0.3993214435,"nll_bias":0.0530405152,"predicted_four_draw_mean_rmse":1.8797078133},{"event":"glucose_lab","n":46,"mean_rmse":2.8174100146,"mean_bias":-0.2798770492,"nll_rmse":0.223822858,"nll_bias":0.1313037926,"predicted_four_draw_mean_rmse":1.9595211744}]

## Run 18: expanded register audit
{"hour":{"('glucose_finger', 'early_with_last')":0.0,"('glucose_finger', 'later_with_last')":13.0366614665,"('glucose_finger', 'no_previous_same_code')":0.77,"('glucose_lab', 'early_with_last')":0.0,"('glucose_lab', 'later_with_last')":13.5618811881,"('glucose_lab', 'no_previous_same_code')":2.8686868687,"('map_arterial', 'early_with_last')":0.0,"('map_arterial', 'later_with_last')":12.5257201646,"('map_arterial', 'no_previous_same_code')":-0.9302325581,"('map_cuff', 'early_with_last')":0.0,"('map_cuff', 'later_with_last')":12.6231221424,"('map_cuff', 'no_previous_same_code')":0.16},"has_last":{"('glucose_finger', 'early_with_last')":1.0,"('glucose_finger', 'later_with_last')":1.0,"('glucose_finger', 'no_previous_same_code')":0.0,"('glucose_lab', 'early_with_last')":1.0,"('glucose_lab', 'later_with_last')":1.0,"('glucose_lab', 'no_previous_same_code')":0.0,"('map_arterial', 'early_with_last')":1.0,"('map_arterial', 'later_with_last')":1.0,"('map_arterial', 'no_previous_same_code')":0.0,"('map_cuff', 'early_with_last')":1.0,"('map_cuff', 'later_with_last')":1.0,"('map_cuff', 'no_previous_same_code')":0.0},"mean_error":{"('glucose_finger', 'early_with_last')":3.2620172923,"('glucose_finger', 'later_with_last')":3.4011175469,"('glucose_finger', 'no_previous_same_code')":-3.7502090028,"('glucose_lab', 'early_with_last')":-0.1829546129,"('glucose_lab', 'later_with_last')":0.082403087,"('glucose_lab', 'no_previous_same_code')":0.0175114366,"('map_arterial', 'early_with_last')":-0.6287206978,"('map_arterial', 'later_with_last')":-0.157678132,"('map_arterial', 'no_previous_same_code')":0.8667915669,"('map_cuff', 'early_with_last')":-0.9905119593,"('map_cuff', 'later_with_last')":-0.2916451057,"('map_cuff', 'no_previous_same_code')":0.3902956651},"nll_error":{"('glucose_finger', 'early_with_last')":0.0843948693,"('glucose_finger', 'later_with_last')":0.0812757134,"('glucose_finger', 'no_previous_same_code')":-0.011269861,"('glucose_lab', 'early_with_last')":-0.1094083951,"('glucose_lab', 'later_with_last')":0.0675890326,"('glucose_lab', 'no_previous_same_code')":-0.0033935062,"('map_arterial', 'early_with_last')":-0.0741055434,"('map_arterial', 'later_with_last')":0.0478211463,"('map_arterial', 'no_previous_same_code')":-0.0778523591,"('map_cuff', 'early_with_last')":-0.0550270855,"('map_cuff', 'later_with_last')":0.0623687087,"('map_cuff', 'no_previous_same_code')":0.0153190197}}
[{"zcol":6,"acol":3,"flags":[0],"age_mode":"seconds_div10","value_activation":"relu","mean_rmse":1.8684446713,"nll_rmse":0.2443487763,"score":2.6014910001},{"zcol":6,"acol":3,"flags":[0],"age_mode":"seconds_div10","value_activation":"gelu","mean_rmse":1.912238264,"nll_rmse":0.2391927505,"score":2.6298165156},{"zcol":6,"acol":3,"flags":[0],"age_mode":"seconds_div10","value_activation":"silu","mean_rmse":1.9303590645,"nll_rmse":0.2631136543,"score":2.7197000273},{"zcol":6,"acol":7,"flags":[0,3],"age_mode":"hours_div10","value_activation":"relu","mean_rmse":2.3047018548,"nll_rmse":0.2447328726,"score":3.0389004728},{"zcol":6,"acol":4,"flags":[0,3],"age_mode":"hours_div10","value_activation":"relu","mean_rmse":2.3141525567,"nll_rmse":0.24308286,"score":3.0434011368}]
Head reconstruction remains provisional until systematic errors are explained.

## Run 19: restart audit
No edits. Best provisional head reconstruction still has mean RMSE 1.87 and NLL RMSE 0.24; recovering the exact architecture and previous evidence before deriving any factual-preserving weight edit.

## Run 22: consolidated evidence and alignment audit
Insulin clean lab threshold estimates approximately -1.55/rate unit; history-adjusted lab estimates approximately -1.97. NE rich-history estimates approximately +3.5 to +3.9/rate unit. Positive persistence is supported, but accumulated rollout targets remain uncertain. No edits; inspecting state/head alignment and exact features.

## Run 24: register and activation audit
MAP discrepancies can be explained by four-draw latent sampling. Searching glucose register flags and boundary conventions. Best config: {'activation': 'gelu', 'gap_mode': 'seconds_div10', 'type_id': 1, 'status_id': 0, 'age_mode': 'seconds_div10', 'zcol': 6, 'acol': 3, 'flags': (0,), 'mean_rmse': 3.579830679304847, 'nll_rmse': 0.40590378333394833, 'score': 4.797542029306692, 'gap_activation': 'gelu', 'prior_activation': 'gelu', 'value_activation': 'gelu'}; bin_side=left
                    bias       mse  expected_sampling_mse
event                                                    
glucose_finger  0.075964  3.536782               3.391510
glucose_lab    -0.232425  3.000851               3.494703
map_arterial   -0.008901  0.559775               0.686899
map_cuff        0.122907  1.171264               1.218299

## Runs 25–26: successful head reconstruction and diagnostic dose probe
All 4660 reading means reconstructed without meaningful channel bias (glucose_finger -0.040, lab -0.091, MAP arterial -0.0002, cuff +0.003).
Head config: GELU for gap/prior/value; query type 1; status 0; last flags (0,), age column 3, value-z column 6; age log1p(seconds)/10; bin side left. Original states: S_reference; baseline head features: head_original_features, 64 prior draws, seed 917.
Run 25 stopped before saving because load_original returns torch.Tensor; run 26 fixed with clone(). Temporary edit ONLY embed.numeric.weight: presence columns 3/5 receive 0.5 times numeric dose/rate columns 2/4. Equivalent rate increments: {'insulin': 0.5, 'norepinephrine': 0.30000000000000004}.
Restore embed.numeric.weight after obtaining probe states. Bins are unchanged; the final correction must be tested with exact dose_edit.
F includes current drug exposures: exclude the target drug's current exposure when estimating its assignment from pre-dose history.

## Run 27: probe collected, original embedding restored
Forward 5 returned 4350 bundle states for the same 100 training stays. Saved as S_probe; removed temporary embed.numeric.weight edit. Recovering architecture and kernel analysis before constructing a lasting correction.

## Run 30: matched head-dose probe
Probe rows align exactly. RMS state differences: out 0.01181, h 0.7450, posterior z 0.04997. Original embedding remains restored. Used validated head_cfg and matched state globals to compute head_probe_features with the same 64 prior draws (seed 917); head_probe_delta_hidden records the controlled hidden-feature changes.

## Run 31: configuration aliasing fixed
Run 30 used a mutable head_cfg that did NOT match the validated reconstruction; its reported probe differences must not be used. Frozen repair_head_cfg explicitly: all GELU, type 1, status 0, flags (0,), age col 3, value-z col 6, log1p(seconds)/10, bin side left. Recomputed repair_original_features and repair_probe_features with matched 64 prior draws (seed 917). Use repair_delta_hidden, not head_probe_delta_hidden, for fitting.

## Run 32: multi-draw, held-out dose-direction diagnostic
Frozen head reconstruction exactly matches saved original features. Collected 8 matched single-prior draws (1701..1708) in probe_draw_store. 70/30 stay split, standardized hidden features plus intercept. Fitted small linear dose-residual decoders using factual-zero and probe-response constraints; no weight edits installed.
[{"probe_weight":0.5,"insulin_factual_rms":0.0046042331,"insulin_probe_gain":0.0003725883,"insulin_probe_rmse":0.2467015587,"norepinephrine_factual_rms":0.0051738076,"norepinephrine_probe_gain":0.003181226,"norepinephrine_probe_rmse":0.1175140045},{"probe_weight":2.0,"insulin_factual_rms":0.017939561,"insulin_probe_gain":0.0014598419,"insulin_probe_rmse":0.246445556,"norepinephrine_factual_rms":0.0177448216,"norepinephrine_probe_gain":0.0108292365,"norepinephrine_probe_rmse":0.1167052642},{"probe_weight":8.0,"insulin_factual_rms":0.066543762,"insulin_probe_gain":0.005473004,"insulin_probe_rmse":0.245568404,"norepinephrine_factual_rms":0.0553342212,"norepinephrine_probe_gain":0.0318937368,"norepinephrine_probe_rmse":0.1147169264},{"probe_weight":32.0,"insulin_factual_rms":0.2205939176,"insulin_probe_gain":0.0184436834,"insulin_probe_rmse":0.243132231,"norepinephrine_factual_rms":0.1650639249,"norepinephrine_probe_gain":0.0839051769,"norepinephrine_probe_rmse":0.1107185954},{"probe_weight":128.0,"insulin_factual_rms":0.6057959684,"insulin_probe_gain":0.0509720594,"insulin_probe_rmse":0.2386374954,"norepinephrine_factual_rms":0.4220075654,"norepinephrine_probe_gain":0.1775896231,"norepinephrine_probe_rmse":0.1056616833},{"probe_weight":512.0,"insulin_factual_rms":1.42675965,"insulin_probe_gain":0.1118989948,"insulin_probe_rmse":0.2356042328,"norepinephrine_factual_rms":0.8728858751,"norepinephrine_probe_gain":0.2797521816,"norepinephrine_probe_rmse":0.1040786225}]

## Run 33: head-only route rejected; within-stay persistence check
Held-out factual-zero decoders cannot recover much probe response: at weight 512 insulin gain 0.112 costs factual RMS 1.427 rate units; NE gain 0.280 costs 0.873. Do not install decoder edits. Next route: amplify action pathway and recalibrate factual predictions using code/last-value transition weights. Differenced IV removes stay intercepts; cross-sensor lag1 and same-sensor lag2 instruments check persistence conditional on externally supported drug coefficients. Interpret with missing-reading selection and instrument-strength caution.
[{"sensor":"glucose_finger","instrument":"other_sensor_lag1","n":46123,"b":-1.55,"rho":-4.9046918397,"cluster_se":2.5152764014,"instrument_x_cov":-2.5765004757},{"sensor":"glucose_finger","instrument":"other_sensor_lag1","n":46123,"b":-1.8,"rho":-4.8978516129,"cluster_se":2.5155655386,"instrument_x_cov":-2.5765004757},{"sensor":"glucose_finger","instrument":"other_sensor_lag1","n":46123,"b":-2.0,"rho":-4.8923794314,"cluster_se":2.5158191282,"instrument_x_cov":-2.5765004757},{"sensor":"glucose_finger","instrument":"same_sensor_lag2","n":211634,"b":-1.55,"rho":7.0994367275,"cluster_se":1.9500454994,"instrument_x_cov":1.7201035657},{"sensor":"glucose_finger","instrument":"same_sensor_lag2","n":211634,"b":-1.8,"rho":7.1033551052,"cluster_se":1.9491779374,"instrument_x_cov":1.7201035657},{"sensor":"glucose_finger","instrument":"same_sensor_lag2","n":211634,"b":-2.0,"rho":7.1064898075,"cluster_se":1.9484893959,"instrument_x_cov":1.7201035657},{"sensor":"glucose_lab","instrument":"other_sensor_lag1","n":2770,"b":-1.55,"rho":-0.4645327882,"cluster_se":0.9978312224,"instrument_x_cov":5.479914329},{"sensor":"glucose_lab","instrument":"other_sensor_lag1","n":2770,"b":-1.8,"rho":-0.5796620075,"cluster_se":1.0520966706,"instrument_x_cov":5.479914329},{"sensor":"glucose_lab","instrument":"other_sensor_lag1","n":2770,"b":-2.0,"rho":-0.671765383,"cluster_se":1.1026415304,"instrument_x_cov":5.479914329},{"sensor":"glucose_lab","instrument":"same_sensor_lag2","n":937,"b":-1.55,"rho":-1.0753677599,"cluster_se":3.3483164007,"instrument_x_cov":3.2403527004},{"sensor":"glucose_lab","instrument":"same_sensor_lag2","n":937,"b":-1.8,"rho":-1.033434824,"cluster_se":3.2946460047,"instrument_x_cov":3.2403527004},{"sensor":"glucose_lab","instrument":"same_sensor_lag2","n":937,"b":-2.0,"rho":-0.9998884754,"cluster_se":3.2539546634,"instrument_x_cov":3.2403527004},{"sensor":"map_arterial","instrument":"other_sensor_lag1","n":77450,"b":3.0,"rho":0.31709045,"cluster_se":0.1947866644,"instrument_x_cov":-0.9592440372},{"sensor":"map_arterial","instrument":"other_sensor_lag1","n":77450,"b":3.6,"rho":0.2655603921,"cluster_se":0.189813322,"instrument_x_cov":-0.9592440372},{"sensor":"map_arterial","instrument":"other_sensor_lag1","n":77450,"b":4.0,"rho":0.2312070202,"cluster_se":0.1866884433,"instrument_x_cov":-0.9592440372},{"sensor":"map_arterial","instrument":"same_sensor_lag2","n":245301,"b":3.0,"rho":-1.0836316812,"cluster_se":0.2942784105,"instrument_x_cov":-0.2672853584},{"sensor":"map_arterial","instrument":"same_sensor_lag2","n":245301,"b":3.6,"rho":-1.2368191822,"cluster_se":0.3181058813,"instrument_x_cov":-0.2672853584},{"sensor":"map_arterial","instrument":"same_sensor_lag2","n":245301,"b":4.0,"rho":-1.3389441829,"cluster_se":0.3345030528,"instrument_x_cov":-0.2672853584},{"sensor":"map_cuff","instrument":"other_sensor_lag1","n":7620,"b":3.0,"rho":0.803761118,"cluster_se":0.7168763323,"instrument_x_cov":-2.2039797465},{"sensor":"map_cuff","instrument":"other_sensor_lag1","n":7620,"b":3.6,"rho":0.7726995729,"cluster_se":0.7052553998,"instrument_x_cov":-2.2039797465},{"sensor":"map_cuff","instrument":"other_sensor_lag1","n":7620,"b":4.0,"rho":0.7519918761,"cluster_se":0.6976004774,"instrument_x_cov":-2.2039797465},{"sensor":"map_cuff","instrument":"same_sensor_lag2","n":218705,"b":3.0,"rho":0.1897217426,"cluster_se":0.1475902608,"instrument_x_cov":-0.9540745251},{"sensor":"map_cuff","instrument":"same_sensor_lag2","n":218705,"b":3.6,"rho":0.1697852257,"cluster_se":0.1461924629,"instrument_x_cov":-0.9540745251},{"sensor":"map_cuff","instrument":"same_sensor_lag2","n":218705,"b":4.0,"rho":0.1564942145,"cluster_se":0.1452857505,"instrument_x_cov":-0.9540745251}]
benchmark_stays=[2020000000, 2020000016, 2020000032, 2020000048, 2020000064, 2020000080, 2020000096, 2020000001, 2020000017, 2020000033, 2020000049, 2020000065, 2020000081, 2020000097, 2020000002, 2020000018, 2020000034, 2020000050, 2020000066, 2020000082, 2020000098, 2020000003, 2020000019, 2020000035, 2020000051, 2020000067, 2020000083, 2020000099, 2020000004, 2020000020, 2020000036, 2020000052, 2020000068, 2020000084, 2020000005, 2020000021, 2020000037, 2020000053, 2020000069, 2020000085, 2020000006, 2020000022, 2020000038, 2020000070, 2020000086, 2020000007, 2020000023, 2020000039, 2020000055, 2020000071, 2020000087, 2020000008, 2020000024, 2020000040, 2020000056, 2020000072, 2020000088, 2020000009, 2020000025, 2020000041, 2020000057, 2020000073, 2020000089, 2020000010, 2020000026, 2020000042, 2020000058, 2020000074, 2020000090, 2020000011, 2020000027, 2020000043, 2020000059, 2020000075, 2020000091, 2020000012, 2020000028, 2020000044, 2020000060, 2020000076, 2020000092, 2020000013, 2020000029, 2020000045, 2020000061, 2020000077, 2020000093, 2020000014, 2020000030, 2020000046, 2020000062, 2020000078, 2020000094, 2020000015, 2020000031, 2020000047, 2020000063, 2020000079, 2020000095]

## Runs 34–37: recovered context
Architecture reread. Clean insulin lab RD about -1.55/unit, rich-history estimate -1.97; NE estimates +3.5 to +3.9/unit. Original 100-stay rollout -3.226/+1.825/+1.373/-0.846. No edits. Persistence must separate stable stay heterogeneity from dynamic propagation; plan a small latent-state fit.

## Run 38: latent-state persistence diagnostic
Fitted own two-sensor Gaussian state models, with stable random stay baselines, measurement noise, missing readings and known hourly drugs. Drug coefficients fixed to record-supported -1.8 and +3.6. This is a statistical diagnostic, not world-model training. Results:
insulin: {'rho': 0.9297708867835622, 'b': -1.8, 'drift': array([0.71915881, 0.89788434]), 'Q': 29.547189040390034, 'R': array([73.44903309,  9.77865272]), 'bias': array([ 0.        , -0.03495557]), 'initial_mean': array([145.30419481, 153.04341247]), 'initial_cov': array([[1182.66875936, 1591.02488485],
       [1591.02488485, 3924.7130875 ]])}
norepinephrine: {'rho': 0.7682574913774728, 'b': 3.6, 'drift': array([ 0.02225472, -0.27350641]), 'Q': 7.533270817867828, 'R': array([ 7.62901515, 33.99916423]), 'bias': array([0.        , 3.08272929]), 'initial_mean': array([74.6301833 , 72.69902581]), 'initial_cov': array([[135.91466218, 162.02437263],
       [162.02437263, 272.29117857]])}

## Run 39: converged persistence and first action candidate
insulin: {'rho': 0.9641085379393141, 'b': -1.8, 'drift': array([0.98483209, 0.78975363]), 'Q': 23.593656044205655, 'R': array([76.75776955, 12.33854852]), 'bias': array([ 0.        , -0.02751273]), 'initial_mean': array([145.0370579 , 153.09561383]), 'initial_cov': array([[1164.91170681, 2119.69166412],
       [2119.69166412, 8135.17530494]])}
norepinephrine: {'rho': 0.8445405942374236, 'b': 3.6, 'drift': array([-0.16167194, -0.27209607]), 'Q': 4.739124223305644, 'R': array([ 9.44581897, 35.72410687]), 'bias': array([0.       , 3.0834658]), 'initial_mean': array([74.71138896, 72.6813748 ]), 'initial_cov': array([[129.69680217, 176.37021955],
       [176.37021955, 346.19154784]])}
Benchmark targets:
                 shift     target
edit                             
insulin_half  1.825296   7.398104
insulin_x2   -3.225785 -14.796209
ne_half      -0.845804  -3.571673
ne_x2         1.372628   7.143345
Installed reversible numeric dose/rate slope amplification 3.0, with per-drug code-embedding offsets analytically preserving the mean recorded atom embedding. Only embed.numeric.weight and embed.code.weight edited. Await exact factual and rollout tests.

## Forward 7: scale-3 numeric candidate
Paired 100-stay tuning rollout: original insulin x2/half -3.226/+1.825, edited -3.247/+1.829; original NE x2/half +1.373/-0.846, edited +2.459/-1.571. Factual pass 6 improved overall NLL by 0.00070. Shared numeric amplification is responsive for NE but almost ineffective for insulin. Need inspect separate pathways; record-derived rollout targets remain estimates, not observed truth.

## Run 45: numeric scale-12 diagnostic
Recovered clean insulin lab RD about -1.55/unit and rich history estimate -1.97/unit. Scale-3 rollout changed NE substantially but not insulin. Installed numeric scale 12, chosen from observed NE response and provisional record target; per-drug code offsets preserve the mean training atom embedding. This remains a reversible diagnostic, awaiting factual and paired rollout tests.

## Runs 46–48: second context recovery
Scale-12 factual NLL on benchmark: original 1.574995, edited 1.576541 (delta +0.001546, within budget). Forward 9: insulin x2/half -3.252/+2.597; NE x2/half +2.521/-3.393. Numeric scaling saturates and cannot alone reach record-supported effects. Inspecting action and decoder pathways before a targeted, reversible regression-derived edit.

## Run 50: posterior-action amplification diagnostic
Reverted scale-12 embed.code/numeric edits; backed them up as numeric12_backup. Only z_post.0.weight edited: action-input columns 384:576 multiplied by 4. This tests whether amplification downstream of action normalization can recover dose sensitivity. No conclusion until exact rollout and factual tests.

## Forward 10 and run 51: bin-path diagnostic
Posterior action x4 had essentially no effect: insulin -3.223/+1.840; NE +1.363/-0.778. Reverted it. Installed dose/rate bin scaling x4 for present bins 1:16 only, leaving absent bin 0 unchanged. Drug code offsets subtract each drug's mean training embedding change. Numeric weights remain original. Awaiting exact rollout and factual tests.

## Runs 52–54; forwards 11–12: reject bin amplification
Dose/rate bin x4 with mean code compensation raised factual NLL by 0.00910 (fails budget). Rollout insulin x2/half -3.032/+1.511; NE x2/half -0.839/-1.798. Reverted all edits. Simple action-embedding amplification is inadequate; investigating a conditional-dose residual direction in original states for a targeted analytical edit.

## Runs 55–58: raw-state route rejected
Recovered prior successful head reconstruction and failed head-only probe edit. Repeated factual-zero/probe-response regression on out, mean_h, all_h and posterior z with held-out stays: no usable tradeoff. Do not amplify a decoded dose residual downstream. Next investigate drug-specific action-encoder directions and factual recalibration.

## Runs 59–65: action encoder reconstruction
Confirmed GELU action encoder and posterior. Action-only bundles use a learned empty-observation encoding, not zero. Single-drug posterior reconstruction error about 3e-5; inferred units insulin=3/2 and NE=4/3 (unit/rate_unit). Checking END atoms and observation encoding before analytical pathway edits. No current edits.

## Runs 66–68: third context recovery
No edits. Observation/END posterior reconstruction is still inaccurate, but prior diagnostics reconstructed single-drug bundles to about 3e-5. Head-only factual-zero dose decoders were previously rejected. Investigating a residual-dose injection into the latent/recurrent pathway, with held-out factual checks before accepting any edit.

## Runs 70–72: END routing resolved
Single-start posterior RMSE 2.66e-5. END atoms belong to the observation encoder, with numeric attributes and units absent, not to the action encoder. This convention gives END-bundle posterior RMSE 4.1e-5. Constructing original and synthetic-dose posterior hidden features; no weights edited.

## Runs 73–78: observation reconstruction checks
Architecture reread; posterior hidden features are 192-dimensional. Measurement unit and categorical-tag searches did not resolve the observation-only posterior mismatch. No weights edited. Checking numerical tables and encoder activations before relying on observation features for a factual-preserving edit.

## Run 81: context recovery
No edits. Drug-only posterior reconstruction remains accurate (~4e-5), but observation reconstruction is inaccurate (~0.14 RMSE). Recovering architecture and helpers before using reconstructed features.

## Run 85: absent-action routing check
       embedding  has_action    n  mean_rmse  median_rmse  max_rmse
        existing       False 2778   0.137688     0.119116  0.832527
        existing        True 1472   0.000042     0.000038  0.000057
absent_rate_unit       False 2778   0.137688     0.119116  0.832527
absent_rate_unit        True 1472   0.000042     0.000038  0.000057

## Runs 93–95: empty-action convention recovered
Saved forward states support the encoded empty action, not zero: no-action posterior RMSE falls from 0.13769 to 0.01344; drug bundles remain about 4e-5. Run 94's diagnostic stopped on normalization key layout, without editing weights.

## Runs 97–101: fourth context recovery and unit correction
Recovered architecture/helpers; no weights edited. Correct measurement unit IDs: glucose_finger=0, glucose_lab=5, map_arterial=map_cuff=2. Rebuilt complete noninitial bundles with END atoms in observations and encoded empty sets in both encoders. Reconstruction summary:
            size      mean       max
has_action                          
False       2778  0.002159  0.119282
True        1472  0.000042  0.000057

## Runs 102–103: dose-residual identifiability in model features
Correct reconstruction on drug bundles (~4e-5); remaining large errors are terminal visit END bundles at hour 25.1667, excluded from fitting. Existing posterior hidden-feature contrast fits have poor held-out factual/dose-response tradeoff, so not installed. Testing uncompressed ctx/e/u inputs:
 probe_weight  insulin_factual_rms  insulin_contrast_gain  insulin_rms_per_gain  norepinephrine_factual_rms  norepinephrine_contrast_gain  norepinephrine_rms_per_gain
         0.03             0.062873               0.183941              0.341810                    0.013736                      0.237258                     0.057895
         0.30             0.209895               0.557639              0.376399                    0.049072                      0.546695                     0.089761
         3.00             0.309315               0.814713              0.379661                    0.101697                      0.805250                     0.126293
        30.00             0.344656               0.925477              0.372409                    0.137159                      0.925320                     0.148229

## Runs 105–110: local intervention reconstruction
Recovered exact-head configuration repair_head_cfg and 4-head recurrent step. Measured finite-difference posterior-z sensitivity at 32 factual drug bundles; results are in latent_J110 and latent_out_J110. No edits installed.

## Run 110 result and run 111
Posterior-z perturbations are essentially ignored by the recurrent model (mean response norms about 0.00057 glucose and 0.000026 MAP per z-unit). Reject posterior amplification. Extracted factual and dose-probe prelude FFN hidden features and E inputs for an analytical direct-state edit. No edits installed.

## Runs 113–120: direct recurrent-input route
No edits. Recovered prelude fits; intercept is FIRST. Affine E residuals can be implemented linearly using the layer-norm invariant (conversion max error 2.4e-5). Exact recurrent convention: 4 heads, memory includes current observation, age in keys and values. Run 119 stopped on read-only npz assignment; run 120 uses a temporary dict. Testing core.b_in injection via finite differences, not gradients.

## Runs 128–132: direct recurrent steering candidate
Recovered finite-difference h-pre-coda steering V128 with near-unit glucose/MAP responses and negligible cross-response. E dose residual is folded into Wdose128 using the layer-normalization invariant (max conversion error about 2.5e-5). Installed only core.b_in.weight:
delta_B = diag(1/softplus(delta_in)) @ V128 @ diag([-5,8]) @ Wdose128.T.
This is a conservative diagnostic, not an accepted final repair. Record-supported immediate effects remain insulin roughly -1.55 to -1.97/rate and NE +3.5 to +3.9/rate; statistical accumulation targets on 100 tuning stays roughly -14.8/+7.4 and +7.1/-3.6, with persistence uncertainty. Next require exact factual budget and paired rollout improvement.

## Runs 133–134: cached-decoder inconsistency
No weights remain edited. The prior Wdose128 failed affine reconstruction on actual E119 too. Reconstructing the decoder directly from fit128 B/mu/sd, with intercept FIRST, then folding its intercept using sum(E/gamma)=sum(beta/gamma). Only a construction matching standardized predictions on all factual E features will be used.

## Run 135: corrected recurrent candidate installed
Resolved the cached-name swap: Wdose128 was RAW slopes, whereas Edecoder128 was already FOLDED. Canonical construction is raw_coef134=B[1:]/sd, affine_b134=B[0]-mu@raw_coef134, Wdose134=raw_coef134+outer(1/gamma,affine_b134/sum(beta/gamma)).
Matches standardized factual predictions exactly and folded predictions within 2.5e-5. Factual residual RMS 0.10747 insulin, 0.03304 NE.
Installed only core.b_in.weight using gains [-5,+8] and V128/softplus(delta_in). Large matrix norm reflects the intercept folded through the layer-normalization invariant; actual factual injection is small. Await exact paired factual test, then rollout if budget holds. Run-132 construction must not be reused.

## Restart after forward 13 / run 136
Current corrected core.b_in edit raises held-out tuning factual NLL from 1.57474 to 1.57640 (+0.00166), within +0.005. Causal rollout improvement is not yet tested. Recovering architecture, record-supported estimates, and evaluation IDs before paired rollout.

## Forward 14: corrected recurrent steering succeeds directionally
99 tuning stays, paired exact rollouts: original insulin x2/half -3.226/+1.825, NE +1.373/-0.846; corrected edited gains [-5,+8] give -9.016/+4.485 and +3.372/-1.557. Factual cost from forward13 +0.00166. Retain this as a viable fallback; audit record-supported accumulation targets before stronger gains.

## Run 139: stronger recurrent candidate
Saved validated gains [-5,+8] matrix as fallback_recurrent139. Reconstruction checked against installed weights. Installed corrected Wdose134/V128 edit at gains [-10,+12], prioritizing insulin. Require exact factual NLL test before accepting or rolling out.

## Forward 15 / run 140: stronger gains rejected on first cohort
Gains [-10,+12] factual NLL 1.58059 vs 1.57474 (+0.00585); exceeds budget. Restored exact fallback_recurrent139 [-5,+8]. Inspecting alternative decoder efficiency and factual shifts before choosing a revised candidate. No rollout spent on failed candidate.

## Runs 142–144: factual-cost attribution
Canonical E decoder Wdose134 and steering V128 recovered; fallback [-5,+8] remains installed. Stronger [-10,+12] costs glucose_finger +0.00815 and map_arterial +0.00815 per sensor. Fallback changes the first glucose reading at hour -1 from 143.99 to 154.79, and stronger to 168.85, despite no previous drug exposure. Investigating omitted initial-bundle zero-dose constraints before accepting a simple stronger gain compromise. Record-based target uncertainty retained.

## Runs 148–150: initial-safe stronger candidate
Recovered the architecture, effect evidence, and constrained decoders. Original fallback [-5,+8] was verified against its canonical construction and saved as initial_safe_fallback150. Installed only core.b_in.weight using constraint_candidates147[(0.3,0.0001)] and gains [-10,+12]. This decoder constrains the initial four slot features to zero dose signal. Require paired factual NLL increase <=0.005 and no initial-reading distortion before rollout. Record-supported accumulation remains uncertain; the previous validated fallback remains available.

## Forward 16–17 / run 151: initial-safe stronger edit validated locally
Candidate key (0.3,0.0001), gains [-10,+12], only core.b_in.weight. 99-stay paired factual NLL 1.57771 vs 1.57474, +0.00297, passes. Paired exact rollouts: insulin x2 -14.181, half +6.780; NE x2 +4.316, half -2.067. Insulin is near the earlier record-supported accumulation estimate; NE remains weaker. Saved exact weights as validated_B151 and decoder/gains as validated_W151/validated_gains151. Need independent-cohort factual validation, target/persistence audit, and possibly a more efficient NE correction before final acceptance.

## Runs 151–152: factual attribution and target audit
Initial/pre-ICU mean shifts <=0.0001, NLL changes zero. Current +0.002976 factual NLL has stay-cluster SE 0.001566 on 99 stays. Sensor costs: finger +0.002123, lab -0.004625, arterial +0.008312, cuff +0.001800. Need new-cohort validation. Earlier latent Gaussian diagnostic gave rho 0.9641 insulin and 0.8445 NE at fixed effects -1.8/+3.6; targets on the first cohort -14.796/+7.398 and +7.143/-3.572. Those estimates rely on the statistical state-model assumptions and must not be treated as known truth.

## Forward 18 / run 153: independent factual validation
Same initial-safe candidate (decoder key (0.3,0.0001), gains [-10,+12], only core.b_in.weight; hash 749033ff8e) on 100 previously untested tuning stays: original NLL 1.63543, edited 1.63777, increase +0.00234 on 4637 readings. Passes +0.005 budget independently. Preserve current weights while auditing record-supported persistence and intervention targets; these are estimates, not truth.

## Run 154: pooled factual validation
Initial-safe [-10,+12] candidate across 199 tuning stays and 9446 readings: paired NLL increase 0.002665, stay-cluster SE 0.001046. Reviewing earlier causal identification and persistence evidence before further changes.

## Runs 157–158: extended record-only persistence fits
NE fixed b=3.6: rho 0.847170 after 48 additional iterations; near numerical convergence. Insulin fixed b=-1.8: rho 0.969626 after 96 additional iterations. These are assumption-dependent record-only estimates. Run157's error was only a reporting lookup of 'hour' instead of 't'. World-model weights unchanged.

## Restart recovery, runs 167–169
Verified that only core.b_in.weight is edited and it exactly matches validated_B151, initial-safe decoder with gains [-10,+12]. Exact reached-hour comparison from run166: per-stay insulin x2/half -12.1599/+5.4133 versus assumption-dependent record targets -12.8893/+6.4447; NE x2/half +4.2690/-2.0001 versus +7.0881/-3.5440. Pooled-hour averages differ from per-stay averages; use the evaluation's per-stay aggregation for final interpretation. No weight changes during recovery. Inspecting decoder efficiency before any stronger NE candidate.

## Runs 170–173: minimum-disturbance NE decoder
Reconstructed factual/probe designs exactly. Zero-factual-target ridge regression, retaining initial-state zero constraints, reduces matched NE RMS from 0.03311 to about 0.02217 in 5-fold stay CV (key 0.03,1e-6). Metadata alignment verified against both drug flags. Installed provisional core.b_in candidate173: insulin unchanged, NE decoder scaled to twice validated doubled-dose probe sensitivity; gains remain [-10,+12]. Exact validated_B151 is the fallback. Require paired factual increase <=0.005 before rollout; record-based persistence targets remain assumption-dependent.

## Run 174: provisional NE enhancement rejected
Forward19 candidate173 factual NLL increase is 0.00621 on the first 99 tuning stays, exceeding the 0.005 budget. Restored validated_B151 exactly. Recovering decoder construction before considering a weaker NE enhancement; no rollout spent on the failed candidate.

## Run 175: reduced minimum-disturbance NE trial
Reconstructed candidate173 exactly within float32 tolerance. Insulin decoder and gains are unchanged. Installed the new NE decoder at 0.75 of candidate173, i.e. 1.5 times validated doubled-dose probe sensitivity, with gains [-10,+12]. Require factual budget before rollout. validated_B151 remains the exact fallback.

## Forward20–21 / run176: reduced NE enhancement passes locally
Only core.b_in.weight edited, hash 4a7d16d8f6. Factual NLL increase +0.00373 on 99 tuning stays. Pooled reached-hour rollout shifts: insulin x2/half -14.134/+6.816; NE x2/half +5.428/-2.333. Saved exact weights as validated_B176 and decoder as validated_W176. Insulin response is stable; NE response increases. Require independent factual validation and record-based target audit before final selection.

## Forward22 / run177: independent and pooled validation
Current candidate factual increase +0.00309 on the independent 100-stay cohort. Across 199 stays/9446 readings, paired increase 0.003422, stay-cluster SE 0.001159. Weights unchanged. Auditing record-based persistence and accumulated-effect identification before any further strengthening.

## Run178: record-effect audit, no weight changes
Reviewing threshold-IV and stay-crossfitted effect estimates together with the fitted latent persistence models. The 199-stay factual budget passes on its point estimate but sampling uncertainty remains; target estimates are not truth.

## Restart / run180: evidence recovery
Weights unchanged. Recovering architecture and record-identification evidence before considering any further NE gain or final acceptance of validated_B176.

## Runs181–183: causal evidence recovered
Independent sensor/history-adjusted record analyses support insulin around -1.8 to -2.0 per rate unit and NE around +3.6. Clean insulin lab threshold estimate is smaller and less precise; finger observation discontinuities warn against using naive RD. Persistence 0.9696/0.8472 comes from an assumption-dependent latent model, not known simulator truth. Current installed edit exactly equals validated_B176. Auditing per-stay readouts and expanding independent factual coverage before refinement.

## Forward23 / run184: factual margin narrows
Current validated_B176 on 100 untouched tuning stays raises factual NLL by +0.00617, exceeding the budget on this cohort. Pooled 299-stay delta 0.004328, cluster SE 0.000924. Do not strengthen current gains without improving generalization or demonstrating a safer candidate. Weights unchanged during audit.

## Run186: restart recovery
Current edit preserved. Rechecking architecture and decoder construction before adding independent training-state constraints; do not strengthen gains given the third-cohort factual-budget failure.

## Forward24 / run187: independent states
Obtained original states for 100 independent training stays (4144 bundles). Current weights unchanged. Reconstruct and validate features before refitting.

## Runs188–189: reconstruction correction
Initial reconstruction missed the two admission END atoms at each stay's terminal bundle and incorrectly treated a tuple as a tensor. Added all recorded END atoms and checked cached encodings, posterior states, and prelude return layout. No weight changes.

## Run190: END-convention audit
Counts alone did not validate reconstruction. Testing earlier best END routing/embedding convention against corrected Efull134 and original posterior states. No weight changes.

## Run191: absent-attribute END reconstruction
Best new-cohort convention empty_all, posterior RMS 0.13043982. Compared corrected old-cohort E against Efull134. No weight changes.

## Run193: reconstruction audit resumed
Run192 unit sweeps did not validate independent-cohort reconstruction. Current validated_B176 remains unchanged. Recovering architecture and previously validated helper conventions before using new states in a fit.

## Runs193–198: independent-state reconstruction fixed
Cached weights and time flow were correct. The remaining error was zero action input for bundles without actions: the forward implementation uses the encoded empty action set. Correcting this gives posterior RMS 3.646e-5 across 4044 noninitial bundles, with maximum per-bundle RMS 1.073e-4. Drug bundles also match at ~4e-5. validated_B176 is unchanged. New training-state features can now be used after confirming the prelude return layout.

## Runs200–202: independent decoder design validated
Current edit still exactly validated_B176. F147/Efull134 reproduce X147 exactly under the saved standardization. The normalized second prelude return supplies independent training features. Comparing pooled zero-factual-target fits at matched current NE doubled-dose probe sensitivity; these residual diagnostics are not factual NLL. No weight changes.

## Run203: stay-held-out decoder comparison
 probe_weight        ridge  old_CV_RMS  new_CV_RMS  pooled_CV_RMS  NE_double_mean  NE_half_mean
       0.0003 1.000000e-07    0.031482    0.027508       0.029611        0.074672     -0.039583
       0.0030 1.000000e-07    0.031513    0.027550       0.029647        0.074731     -0.039742
       0.0030 1.000000e-06    0.031645    0.027513       0.029702        0.074766     -0.040470
       0.0100 1.000000e-06    0.031740    0.027641       0.029812        0.074881     -0.040810
       0.0300 1.000000e-06    0.032059    0.028065       0.030178        0.075083     -0.041547
Weights remain validated_B176. Decoder residuals are only a proxy for factual disturbance, not a substitute for NLL.

## Run204: provisional independent-state NE refit
Installed only core.b_in.weight: insulin decoder unchanged; NE decoder from pooled old/new factual covariance, key (0.003,1e-7), matched to validated_B176 doubled-dose probe mean. Validated_B176 is the exact fallback. Must test factual NLL on the previously difficult third tuning cohort before acceptance.

## Forward25 / run205: NE refit helps but is not sufficient
Difficult-cohort factual delta 0.00557747; still above 0.005 locally. Investigating zero-factual-target insulin fits at matched existing doubled-dose probe sensitivity, without weight changes in this run. Held-out diagnostics:
    pw        ridge  old_CV_RMS  new_CV_RMS   double      half
0.0003 1.000000e-07    0.091973    0.086946 0.331540 -0.118723
0.0030 1.000000e-07    0.091989    0.086980 0.331607 -0.118946
0.0030 1.000000e-06    0.091747    0.086704 0.331614 -0.119333
0.0100 1.000000e-06    0.091791    0.086791 0.331764 -0.119845
0.0300 1.000000e-06    0.091973    0.087102 0.332092 -0.121077

## Run206: insulin sensitivity constraints
Unconstrained zero-factual fits reduce insulin residuals but weaken half-dose sensitivity and increase cross-drug sensitivity. Comparing small constrained regressions that preserve or proportionally adjust half-dose sensitivity and preserve or remove cross-decoder response. No claim that proxy residual improvements establish causal accuracy; rollouts and record evidence remain required. Weights unchanged.

## Run207: restart recovery
No weight changes. Recovering architecture, record-effect evidence, and constrained-decoder definitions before a direct factual test. Run206 residual reductions remain proxy evidence only.

## Run208: compact candidate recovery
Architecture read; no edits. Run207 output was truncated before candidate inventory. Inspecting recent saved objects only.

## Run210–211: failed mapping, no installation
Run210 stopped before save_edit because assumed X147 standardization of raw Efull134 failed (max error 9.20). Current edit remains candidate_B204. Recovering exact cached feature design/conversion rather than using an unvalidated map.

## Run212: constrained insulin trial installed
Recovered intercept-FIRST design convention. Verified exact cached standardization, LayerNorm affine folding and rank-two factorization. Installed preserve-half/preserve-cross insulin fit (0.003,1e-6), retaining candidate_W204 NE decoder. Only core.b_in.weight edited. Exact fallback: fallback_B212. Candidate requires paired factual NLL and rollout validation.


## Forward26 / runs213–214: constrained trial needs more margin
Difficult-cohort factual NLL delta +0.0050932033, stay-cluster SE 0.0014217141. By-event deltas: finger +0.004080, lab +0.000902, arterial +0.004678, cuff +0.007451. Improves on candidate_B204 but remains above 0.005 on this cohort. No weight changes; recovering candidates and record-based targets.

## Run215: compact recovery
Run214 output clipped before kernel inventory. Current weights retained as current_B215; original as original_B215. Compact candidate and target inventory only; no weight changes.

## Run216: effect and validation inventory
Current candidate remains the rank-two constrained insulin / pooled NE edit. Recovered record-effect and persistence objects and identified untouched tuning stays. No weight changes.

## Forward27 / run217: untouched factual validation and target alignment
Current candidate factual summaries: {'readings': 9475, 'stays': 200, 'original': np.float64(1.6175059208443274), 'edited': np.float64(1.6201375725593667), 'delta': np.float64(0.0026316517150395775), 'cluster_se': np.float64(0.0010610393749883006)}. Record targets use b_insulin=-1.8, rho=0.9696261, b_NE=3.6, rho_NE=0.8471704; these remain identification/model assumptions, not simulator truth. Audited pooled-hour versus equal-stay weighting and matched prior rollout hours. No weight changes.

## Forward28 / run218: untouched rollout and exact factor recovery
Current candidate improves all four responses over original on untouched stays. Compared responses to record targets at exactly the reached measurement hours. Recovered gains [0.9999999999691687, 1.000000000073213] and separate delta_ins218/delta_ne218, factor error 9.45e-07. No weight changes. Any strengthening requires direct factual validation.

## Run219: controlled gain trial
Installed candidate_B219 = original + 1.20*delta_ins218 + 1.35*delta_ne218, only core.b_in.weight. Exact fallback_B219 is candidate_B212 (hash 973acf3505). Motivation: on untouched matched reached hours current shifts are -11.26594/+4.61005 versus record targets -14.87714/+7.43857; NE +5.55432/-2.54535 versus +8.69952/-4.34976. Targets assume the fitted latent persistence and identified drug slopes. This stronger trial is NOT accepted; require paired factual budget and rollout validation.

## Run220: stronger-trial validation recovery
No weight changes. Recovering architecture, exact difficult-cohort IDs, and cached evidence before paired factual testing. candidate_B219 remains provisional; fallback_B219 is preserved.

## Forward29 / Run221: stronger trial fails difficult cohort
Paired difficult-cohort factual summary: {'forward': 29, 'stays': 100, 'readings': 4664, 'min_id': 2020000199, 'max_id': 2020000298, 'delta': 0.008066402229845627, 'cluster_se': 0.0018793886854765963}. candidate_B219 is not accepted. Recovering complete tuning coverage and prior evidence before choosing gain reduction; weights unchanged and fallback_B219 remains exact.

## Forwards29–32 / Run222: complete stronger-trial factual failure
All 400 tuning stays: {'stays': 400, 'readings': 18977, 'original': 1.6108130631817463, 'edited': 1.6159265690045843, 'delta': 0.0051135058228381726, 'cluster_se': 0.0010240357303300742, 'upper_95_one_sided': 0.006798044599231144}. The 1.20 insulin / 1.35 NE trial is not accepted. Next measure its matched untouched-cohort dose responses to inform a reduced-NE candidate; no weight changes in this run.

## Forward33 / Run223: reduced-NE controlled trial
Strong trial dose shifts on matched untouched hours: -13.111/+5.246 glucose; +7.078/-3.230 MAP. Strong trial fails full tuning factual budget (+0.00511351). Installed candidate_B223 = original + 1.20*delta_ins218 + 1.05*delta_ne218, only core.b_in.weight. Retains main insulin gain while reducing MAP disturbance. NOT accepted until full factual and dose validation. fallback_B219 (candidate_B212) remains exact.

## Forwards34–37 / Run224: complete reduced-NE factual validation
Current edit unchanged: original + 1.20*delta_ins218 + 1.05*delta_ne218; only core.b_in.weight.
{"stays": 400, "readings": 18977, "original": 1.6103412288559835, "edited": 1.6136449544185065, "delta": 0.0033037255625230537, "cluster_se": 0.0008822539730740794, "upper_95_one_sided": 0.004755033348229914}
Need matched dose rollout validation; training-record effect targets remain estimates, not revealed simulator truth.

## Forward38 / Run225: reduced-NE trial validated and frozen
Validated fallback validated_B225: only core.b_in.weight; gains 1.20 insulin / 1.05 NE.
Four matched rollout shifts: {"insulin_half": 5.338694567404427, "insulin_x2": -12.99491046277666, "ne_half": -2.784615853658537, "ne_x2": 5.7445625}
Full tuning factual delta 0.00330373, clustered SE 0.000882254, one-sided 95% upper 0.00475503.
Further changes require factual and rollout validation; no weights changed in this run.

## Runs226–227: half-dose decoder blend diagnostics
Run226 inventory error occurred before any edits. No weight changes in either run.
Verified standardization and LayerNorm-to-weight mapping.
 linear_half_fraction  old_factual_RMS  insulin_double  insulin_half  NE_double_cross  mapping_error
                 0.00         0.083019        0.335548     -0.132855         0.097055   7.840183e-06
                 0.25         0.084013        0.335548     -0.141585         0.097055   5.946201e-06
                 0.50         0.085945        0.335548     -0.150314         0.097055   4.052219e-06
                 0.75         0.088753        0.335548     -0.159044         0.097055   2.158237e-06
                 1.00         0.092358        0.335548     -0.167774         0.097055   2.642548e-07
Residual RMS is only a disturbance proxy; any trial needs direct paired factual NLL and dose rollouts.

## Runs228–231: restart verification
Architecture and causal-evidence notes recovered. Drug slopes and latent persistence remain record-based estimates with identification assumptions. No weight changes. Current validated_B225 reconstruction checks: {'current_vs_validated': np.float64(0.0), 'insulin_factor_error': np.float64(3.006803694916016e-10), 'NE_factor_error': np.float64(1.983117670079082e-09), 'mapped_base_error': np.float64(8.870681966755001e-14), 'validated_gain_error': np.float64(3.301189497761925e-06)}. Preparing a controlled half-dose decoder trial.

## Run232: isolated half-dose blend trial installed
Only core.b_in.weight edited. Candidate_B232 uses 50% linear-half decoder blend, retaining insulin gain 1.20 and NE gain 1.05. Mapping error 4.05e-06. Exact accepted fallback is validated_B225 / fallback_B232. Trial is provisional: require paired factual NLL and exact dose rollouts.

## Run233: restarted half-dose-blend factual audit
Only core.b_in.weight remains edited; no weights changed in this run.
Forwards39–42 paired factual summary: {"forwards": [39, 40, 41, 42], "stays": 400, "readings": 18977, "original": 1.6103411972387627, "edited": 1.6138125889234336, "delta": 0.0034713916846709166, "cluster_se": 0.0009053962524777084, "upper_95_one_sided": 0.004960636332143885}
Accepted fallback forwards34–37: {"forwards": [34, 35, 36, 37], "stays": 400, "readings": 18977, "original": 1.6103412288559835, "edited": 1.6136449544185065, "delta": 0.0033037255625230537, "cluster_se": 0.0008822539730740794, "upper_95_one_sided": 0.004754904539149845}
Dose-blend trial remains provisional until exact rollout validation.

## Run234: rollout-cohort and causal-evidence recovery
No weight changes. Half-dose blend factual delta is 0.00347139, one-sided 95% upper 0.00496064; fallback delta is 0.00330373.
Recovered 99 matched rollout stays from forward38 for direct response comparison.

## Forward43 / Run235: half-dose-blend rollout comparison
[{"edit":"insulin_half","fallback":5.3386945674,"blend":5.5372092555,"difference":0.1985146881,"paired_stay_se":0.1702139203,"equal_stay_blend":4.4292739589},{"edit":"insulin_x2","fallback":-12.9949104628,"blend":-13.1459040241,"difference":-0.1509935614,"paired_stay_se":0.2159453205,"equal_stay_blend":-11.2810891603},{"edit":"ne_half","fallback":-2.7846158537,"blend":-2.6234759146,"difference":0.161139939,"paired_stay_se":0.0619548504,"equal_stay_blend":-2.5625114838},{"edit":"ne_x2","fallback":5.7445625,"blend":5.7246381098,"difference":-0.0199243902,"paired_stay_se":0.0717695659,"equal_stay_blend":5.6401008934}]
Current candidate_B232 remains provisional; no weights changed. Factual cost +0.00347139. Insulin benefit is modest; NE half worsens slightly.

## Run236: gain-only versus half-blend diagnostics
No weights changed. Half-blend insulin improvements have paired stay SEs larger than their means; NE-half worsening is clearer.
[{"candidate":"base","insulin_gain":1.0,"NE_gain":1.0,"stays":200,"paired_delta_common_stays":0.0026316517},{"candidate":"strong","insulin_gain":1.2,"NE_gain":1.35,"stays":200,"paired_delta_common_stays":0.0049395251},{"candidate":"accepted","insulin_gain":1.2,"NE_gain":1.05,"stays":200,"paired_delta_common_stays":0.0032247282},{"candidate":"half_blend","insulin_gain":1.2,"NE_gain":1.05,"stays":200,"paired_delta_common_stays":0.0032167071}]
Gain-only diagnostics are planning proxies; any new candidate requires full paired factual validation and the exact rollout.

## Run237: final gain-only controlled trial installed
Only core.b_in.weight edited: original + 1.40*delta_ins218 + 1.00*delta_ne218. Half-dose blend removed.
Exact accepted fallback is fallback_B237 = validated_B225. candidate_B232 remains saved but is not the fallback.
Three-point additive quadratic common-cohort factual proxy: 0.0033930105540897083; this is extrapolation, not acceptance evidence.
Require all-400 paired factual validation and exact matched rollouts. Record-based drug slopes and persistence remain assumption-dependent estimates.


## Run238: final gain-only trial recovery
Forward44 difficult-cohort factual increase is approximately +0.00669; this cohort alone does not establish the full-tuning result. Recovered architecture, record schema, earlier causal evidence, persistent candidate variables and factual cohort IDs. No weights changed. Candidate_B237 remains provisional; fallback_B237 remains the accepted fallback.


## Run239: compact final-validation cohort recovery
Recovered 400 reference tuning stays and the 100 already evaluated in forward44; stored three disjoint remaining groups of 100. Inspected current candidate and accepted-fallback fingerprints. No weights changed. Next complete paired factual validation before accepting the final gain-only trial.


## Run240: complete final gain-only factual validation
Current candidate_B237 unchanged; only core.b_in.weight edited.
Forwards44–47: {"stays": 400, "readings": 18977, "original": 1.61081307372082, "edited": 1.6145614586077883, "delta": 0.0037483848869684345, "cluster_se": 0.0009381402584046192, "upper_95_one_sided": 0.005291488293594463}
Accepted fallback forwards34–37: {"stays": 400, "readings": 18977, "original": 1.6103412288559835, "edited": 1.6136449544185065, "delta": 0.0033037255625230537, "cluster_se": 0.0008822539730740794, "upper_95_one_sided": 0.004754904210026299}
Recovered matched rollout stay IDs. Candidate remains provisional pending exact dose-rollout comparison; estimates from the records remain assumption-dependent, not revealed simulator truth.


## Run241: duplicate-result recovery
Forward48 dose files absent/incomplete; no new rollout evidence.
No weights changed; candidate_B237 remains provisional.

## Run242: final-decision evidence recovery
No weights changed. Forward48 has not run. Recovering architecture, record-based causal estimates, and the matched rollout cohort before spending the three remaining forward passes. candidate_B237 remains provisional; accepted fallback_B237 remains available.

## Run243: compact causal-target and rollout recovery
Recovered 99 matched rollout stays from forward38. No weights changed. Next compare candidate_B237 on these stays using the exact dose-edit rollout; do not treat model predictions or observational effect estimates as revealed simulator truth.

## Forward48 / Run244: stronger insulin rollout validated
Only core.b_in.weight remains edited; candidate_B237 unchanged (insulin gain 1.40, NE gain 1.00). Matched reached-hour shifts: {"insulin_half": 5.943390140845071, "insulin_x2": -14.896366599597586, "ne_half": -2.5891978658536585, "ne_x2": 5.6116676829268295}. Full tuning factual delta 0.003748385, clustered SE 0.000938140, one-sided 95% upper 0.005291626.
Main insulin response now agrees with the assumption-dependent record target; NE and insulin-half magnitudes remain underestimated. Selected 200 fresh training stays for two final paired factual checks. These are training-distribution checks, not new held-out tuning data. fallback_B237 remains available if safety evidence deteriorates.

## Run245: final audit and submission decision
Retain candidate_B237 exactly as tested in forward48 and factual forwards44–47,49–50. Only core.b_in.weight is edited: original + 1.40*delta_ins218 + 1.00*delta_ne218. No half-dose decoder blend, no gradient training, and no further weight changes in this run.
SHA256 of submitted float32 tensor: ef90724d02ecceff7198dbbb10594a02686edc7c75f0264170c933c543a78caf
Factual audit: {"held_out_tuning": {"stays": 400, "readings": 18977, "original_nll": 1.61081307372082, "edited_nll": 1.6145614586077883, "delta": 0.0037483848869684345, "stay_cluster_se": 0.0009381402584046191, "descriptive_one_sided_95_upper": 0.005291625612044033}, "fresh_training_checks": {"stays": 200, "readings": 9487, "original_nll": 1.6487808474754928, "edited_nll": 1.6512736059871402, "delta": 0.002492758511647517, "stay_cluster_se": 0.0011018215299804688, "descriptive_one_sided_95_upper": 0.0043052549284653885}, "pooled_descriptive": {"stays": 600, "readings": 28464, "original_nll": 1.6234676644182124, "edited_nll": 1.6267975512928612, "delta": 0.0033298868746486784, "stay_cluster_se": 0.000725584019746213, "descriptive_one_sided_95_upper": 0.004523472587131198}}
Matched 99-stay tuning rollout readouts: {"insulin_half": 5.943390140845071, "insulin_x2": -14.896366599597586, "ne_half": -2.5891978658536585, "ne_x2": 5.6116676829268295}
Main insulin response closely matches the training-record-derived target on this cohort. Insulin-half and NE magnitudes remain smaller than their estimated targets. Drug slopes and accumulated effects depend on identification and persistence assumptions; none are revealed simulator truth. Additional training checks support factual stability but are not independent held-out tuning evidence. Held-out tuning empirical factual increase is below +0.005; unseen-stay performance is not guaranteed.
All 50 forward passes used; 245 Python runs used. Final decision is the stronger tested edit, not fallback_B237.
