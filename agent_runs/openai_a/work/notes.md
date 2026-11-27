# Weight-repair notebook

## Initial inspection
- Python runs: 1. Forward requests: 0.
- No drug effects assumed; all effect estimates must come from these simulated records.
- First inspect architecture, event ordering, dose encoding, and available histories.
- Then estimate dose effects, measure original rollouts/factual fit, and test targeted weight edits with the factual NLL constraint.

## Architecture inspected; Python 2, forward 0
- Drug atoms use encode_action, enter the latent posterior and recurrent prelude.
- Reading distributions have 17-bin heads plus code/last-bin transition offsets.
- Dose edits alter both numeric scores and dose/rate bins; rollouts feed generated readings back.
- Need record-derived effects and baseline forward results before choosing edits.

## Record transition analysis; Python 3, forward 0
- 30,000 training stays, 400 tuning stays; readings hourly and drugs at hour + 0.0833.
- Built an hourly panel; missing infusion records mean no recorded infusion.
- Compared raw, current-value, history-adjusted and stay-fixed-effect one-hour changes.
{
  "('glucose_finger', 'insulin', 'raw')": {
    "beta": -2.2504397036455344,
    "se": 0.026964113200911532,
    "n": 298450,
    "dose_resid_sd": 1.082455066392304
  },
  "('glucose_finger', 'insulin', 'current')": {
    "beta": -1.939694829612264,
    "se": 0.04568529082750635,
    "n": 298450,
    "dose_resid_sd": 0.635108505538118
  },
  "('glucose_finger', 'insulin', 'history')": {
    "beta": -1.9331334225345693,
    "se": 0.040626786035861666,
    "n": 298450,
    "dose_resid_sd": 0.6341789754358419
  },
  "('glucose_finger', 'insulin', 'history+stayFE')": {
    "beta": -2.0182153553412503,
    "se": 0.040508470697558384,
    "n": 298450,
    "dose_resid_sd": 0.606944671817386
  },
  "('map_arterial', 'norepinephrine', 'raw')": {
    "beta": 0.33512333432011987,
    "se": 0.014602284607062997,
    "n": 312563,
    "dose_resid_sd": 0.34169129921596825
  },
  "('map_arterial', 'norepinephrine', 'current')": {
    "beta": -2.207418302293895,
    "se": 0.04339113183787943,
    "n": 312563,
    "dose_resid_sd": 0.23373813222151651
  },
  "('map_arterial', 'norepinephrine', 'history')": {
    "beta": 0.9721922968719852,
    "se": 0.08572714800470202,
    "n": 312563,
    "dose_resid_sd": 0.10388666692459317
  },
  "('map_arterial', 'norepinephrine', 'history+stayFE')": {
    "beta": 1.9472917386017898,
    "se": 0.0887710568611179,
    "n": 312563,
    "dose_resid_sd": 0.09716625430945373
  }
}

## Original rollout and identification checks; Python 4, forward 1
- Original first 100 tuning stays: insulin x2 -1.847, half +0.777; NE x2 -0.498, half -0.600.
- Sensor disagreement is only a candidate instrument; validity and relevance must be checked, not assumed.
{
  "('insulin', 'glucose_finger', 'history')": {
    "beta": -2.3649672210093593,
    "se": 0.2966072448330681,
    "first_stage": 0.022546708411775182,
    "first_stage_t": 65.79471543842729,
    "n": 59611
  },
  "('insulin', 'glucose_finger', 'past_difference')": {
    "beta": -2.4030110145183943,
    "se": 0.296995873105088,
    "first_stage": 0.02253050217525474,
    "first_stage_t": 65.9033137514767,
    "n": 59611
  },
  "('insulin', 'glucose_finger', 'stayFE')": {
    "beta": -4.633653857866378,
    "se": 0.32319976462186084,
    "first_stage": 0.023609977158208926,
    "first_stage_t": 56.920556473364606,
    "n": 59611
  },
  "('insulin', 'glucose_lab', 'history')": {
    "beta": -1.6841133458408586,
    "se": 0.4993785896440475,
    "first_stage": 0.024098411603330767,
    "first_stage_t": 34.65836012846078,
    "n": 15729
  },
  "('insulin', 'glucose_lab', 'past_difference')": {
    "beta": -1.7098169631722566,
    "se": 0.49571479741038577,
    "first_stage": 0.024088881089054243,
    "first_stage_t": 34.682785128203996,
    "n": 15729
  },
  "('insulin', 'glucose_lab', 'stayFE')": {
    "beta": 0.6604909236239173,
    "se": 0.531745510423111,
    "first_stage": 0.027494693802790897,
    "first_stage_t": 23.486841143670325,
    "n": 15729
  },
  "('norepinephrine', 'map_arterial', 'history')": {
    "beta": -18.770179837459043,
    "se": 10.725044751264742,
    "first_stage": 0.00023924704982281593,
    "first_stage_t": 4.630621380691009,
    "n": 93413
  },
  "('norepinephrine', 'map_arterial', 'past_difference')": {
    "beta": -18.46746024463273,
    "se": 10.79598180969503,
    "first_stage": 0.00023698386029265031,
    "first_stage_t": 4.589330754374463,
    "n": 93413
  },
  "('norepinephrine', 'map_arterial', 'stayFE')": {
    "beta": 45.967013136779165,
    "se": 17.626632496359736,
    "first_stage": 0.0001960059721602588,
    "first_stage_t": 3.583611528216762,
    "n": 93413
  },
  "('norepinephrine', 'map_cuff', 'history')": {
    "beta": -44.76692008065325,
    "se": 44.17355258218645,
    "first_stage": 0.0001653959279835562,
    "first_stage_t": 1.8429218212513345,
    "n": 29358
  },
  "('norepinephrine', 'map_cuff', 'past_difference')": {
    "beta": -44.445574103820974,
    "se": 44.36183449302977,
    "first_stage": 0.00016465813087278928,
    "first_stage_t": 1.836249722244463,
    "n": 29358
  },
  "('norepinephrine', 'map_cuff', 'stayFE')": {
    "beta": -670.8096489513821,
    "se": 406.4526866401043,
    "first_stage": 0.00019082169114508174,
    "first_stage_t": 1.659997261215908,
    "n": 29358
  }
}

## Factual baseline and dynamic identification; Python 5, forward 2
- Original factual value NLL = 1.63461 on first 100 tuning stays, 4751 readings.
- NE sensor-difference first stage is weak; do not use its point estimates.
- Tested replicate-sensor corrections and lag-instrumented first differences; each relies on explicit measurement-noise/innovation assumptions.
{
  "('map_arterial', 'norepinephrine', 'basic')": {
    "persistence": 0.9982457546092935,
    "effect": 0.25976722831925286,
    "se_persistence": 0.0018481345452363164,
    "se_effect": 0.07259737043369274,
    "n": 93413
  },
  "('map_arterial', 'norepinephrine', 'history')": {
    "persistence": 0.9158430901733471,
    "effect": 3.2667303775592043,
    "se_persistence": 0.009083782933650895,
    "se_effect": 0.1636034525210515,
    "n": 93413
  },
  "('map_arterial', 'norepinephrine', 'stayFE')": {
    "persistence": 0.7479846009200942,
    "effect": 3.610685143623699,
    "se_persistence": 0.012280238981277945,
    "se_effect": 0.17246200287179653,
    "n": 93413
  },
  "('map_arterial', 'norepinephrine', 'history_interior')": {
    "persistence": 0.9574567466723566,
    "effect": 3.7778390489116442,
    "se_persistence": 0.013217293064316842,
    "se_effect": 0.1959188852393693,
    "n": 36923
  },
  "('map_cuff', 'norepinephrine', 'basic')": {
    "persistence": 1.002179540391177,
    "effect": 0.5189648230656696,
    "se_persistence": 0.005137497349731058,
    "se_effect": 0.21187411701185913,
    "n": 29358
  },
  "('map_cuff', 'norepinephrine', 'history')": {
    "persistence": 0.9167489009317886,
    "effect": 3.0586191243152263,
    "se_persistence": 0.019627357899411266,
    "se_effect": 0.48172997418062946,
    "n": 29358
  },
  "('map_cuff', 'norepinephrine', 'stayFE')": {
    "persistence": 0.6963059214339552,
    "effect": 3.607914129367453,
    "se_persistence": 0.030053332950615613,
    "se_effect": 0.6219210550781281,
    "n": 29358
  },
  "('map_cuff', 'norepinephrine', 'history_interior')": {
    "persistence": 0.9918898329266252,
    "effect": 3.5777343990133375,
    "se_persistence": 0.030677518752430022,
    "se_effect": 0.6032396349652878,
    "n": 11483
  },
  "('glucose_finger', 'insulin', 'basic')": {
    "persistence": 1.1815187157710842,
    "effect": -6.880464680589379,
    "se_persistence": 0.0034121325995547193,
    "se_effect": 0.1123229737480663,
    "n": 59611
  },
  "('glucose_finger', 'insulin', 'history')": {
    "persistence": 1.139344433365212,
    "effect": -6.754836344624256,
    "se_persistence": 0.0072183527919846245,
    "se_effect": 0.11495405608921397,
    "n": 59611
  },
  "('glucose_finger', 'insulin', 'stayFE')": {
    "persistence": 1.0449899248496881,
    "effect": -6.911335202615895,
    "se_persistence": 0.011256504878414144,
    "se_effect": 0.14134484464766772,
    "n": 59611
  },
  "('glucose_finger', 'insulin', 'history_interior')": {
    "persistence": 1.204625042122073,
    "effect": -7.614864594436871,
    "se_persistence": 0.009406444533304092,
    "se_effect": 0.1357291472186261,
    "n": 50425
  },
  "('map_arterial', 'norepinephrine', 'differenceIV')": {
    "persistence": 0.55224068979847,
    "effect": 9.765249075734024,
    "se_persistence": 0.2031452415734852,
    "se_effect": 3.822496587192125,
    "n": 6907
  },
  "('map_cuff', 'norepinephrine', 'differenceIV')": {
    "persistence": 0.09308982432128969,
    "effect": -1.4392193870006853,
    "se_persistence": 0.2550564286432103,
    "se_effect": 4.71176168711688,
    "n": 6890
  },
  "('glucose_finger', 'insulin', 'differenceIV')": {
    "persistence": 0.6225459139836054,
    "effect": -2.750197720426045,
    "se_persistence": 0.5273550439334205,
    "se_effect": 3.422934062742969,
    "n": 1885
  }
}

## More efficient dynamic checks; Python 6, forward 2
- Replicate-sensor NE estimates are positive but rely on dose/error independence.
- Insulin replicate-sensor level estimates are invalid if dosing uses finger noise; the negative sensor-difference IV is more defensible for insulin.
- Enlarged first-difference samples using own-sensor instruments at lags >=2, which avoid overlapping current measurement errors.
{
  "('map_arterial', 'admission')": {
    "coefficients": [
      0.9825267853113245,
      0.5165933079110422,
      0.029677058825007888
    ],
    "se": [
      0.006061144800747398,
      0.1349655273885699,
      0.006965149985368135
    ],
    "n": 25283
  },
  "('map_cuff', 'admission')": {
    "coefficients": [
      0.9736397009521174,
      0.4870228377565288,
      0.04526212305580476
    ],
    "se": [
      0.014298533258265632,
      0.39942899381697966,
      0.01616727199739762
    ],
    "n": 7995
  },
  "('glucose_finger', '(2, 3)', False)": {
    "persistence": 1.0636624817043296,
    "effect": -20.50967480255291,
    "se_persistence": 0.6011473477865716,
    "se_effect": 14.10406179320535,
    "n": 187065,
    "condition": 4278.111074394605
  },
  "('glucose_finger', '(2, 3)', True)": {
    "persistence": 1.6853771922182748,
    "effect": -28.31022511468185,
    "se_persistence": 0.5252171736929621,
    "se_effect": 16.30187699616657,
    "n": 151972,
    "condition": 4146.207612178202
  },
  "('glucose_finger', '(2, 3, 4)', False)": {
    "persistence": 1.7622489453213448,
    "effect": -32.365447466632254,
    "se_persistence": 0.4640301018640856,
    "se_effect": 14.770938349276275,
    "n": 170139,
    "condition": 3163.075474062332
  },
  "('glucose_finger', '(2, 3, 4)', True)": {
    "persistence": 1.9119603835647734,
    "effect": -30.143412757950454,
    "se_persistence": 0.457167597898931,
    "se_effect": 14.832568278064965,
    "n": 140261,
    "condition": 4231.202488567382
  },
  "('map_arterial', '(2, 3)', False)": {
    "persistence": 0.56601743744358,
    "effect": 2.7506419331801824,
    "se_persistence": 0.07126114745990335,
    "se_effect": 0.3355457823113206,
    "n": 223342,
    "condition": 112.63924108944546
  },
  "('map_arterial', '(2, 3)', True)": {
    "persistence": 0.5766677284650158,
    "effect": 2.1253207072251303,
    "se_persistence": 0.07454238978195318,
    "se_effect": 0.6283698960977526,
    "n": 212970,
    "condition": 268.15917645613507
  },
  "('map_arterial', '(2, 3, 4)', False)": {
    "persistence": 0.49069634893724623,
    "effect": 3.0574728886354428,
    "se_persistence": 0.0461088749782071,
    "se_effect": 0.3461294557797029,
    "n": 201643,
    "condition": 163.09105288561636
  },
  "('map_arterial', '(2, 3, 4)', True)": {
    "persistence": 0.4968152626735331,
    "effect": 2.366824568449006,
    "se_persistence": 0.04891121747249014,
    "se_effect": 0.5944101362422026,
    "n": 192271,
    "condition": 354.87191394661596
  },
  "('map_cuff', '(2, 3)', False)": {
    "persistence": 0.6268160682431204,
    "effect": 6.790448718802579,
    "se_persistence": 0.23618181454957313,
    "se_effect": 2.3593492092276183,
    "n": 183795,
    "condition": 190.83738903656888
  },
  "('map_cuff', '(2, 3)', True)": {
    "persistence": 0.7342361832419435,
    "effect": 12.700388178520777,
    "se_persistence": 0.23291547909078456,
    "se_effect": 4.609211211457707,
    "n": 165686,
    "condition": 576.8037490840541
  },
  "('map_cuff', '(2, 3, 4)', False)": {
    "persistence": 0.5018208037686507,
    "effect": 6.631520491739785,
    "se_persistence": 0.1459720754302214,
    "se_effect": 1.7341403751240727,
    "n": 156601,
    "condition": 273.0146298294646
  },
  "('map_cuff', '(2, 3, 4)', True)": {
    "persistence": 0.5759442765765651,
    "effect": 10.79062518132017,
    "se_persistence": 0.17155737451875266,
    "se_effect": 3.286390891868858,
    "n": 141406,
    "condition": 645.7722117251755
  }
}

## Restart: Python 7, forward 2
- No edits yet. Insulin lag-IV is weak and unstable; NE estimates depend strongly on assumptions.
- Recovering architecture, record timing, and earlier diagnostics before selecting a causal estimator or weight edit.

## Sensor-contrast local projections; Python 9, forward 2
- Reimplemented IV on future reading levels, controlling only pre-treatment history.
- Pre-treatment placebo horizons test independence; weak first stages remain disqualifying.
{
  "('insulin', -2, False)": {
    "n": 57248,
    "beta": -0.038767269782738374,
    "se": 0.6338392802791879,
    "first_stage": 0.024831879550323475,
    "first_stage_t": 46.31225775687944
  },
  "('insulin', -1, False)": {
    "n": 61822,
    "beta": -0.10167365879195693,
    "se": 0.6044600978809811,
    "first_stage": 0.0250430044370933,
    "first_stage_t": 48.56454221088735
  },
  "('insulin', 1, False)": {
    "n": 61734,
    "beta": -9.265448953253808,
    "se": 1.0477733869485997,
    "first_stage": 0.01858921651030461,
    "first_stage_t": 35.99484856927351
  },
  "('insulin', 2, False)": {
    "n": 57208,
    "beta": -3.307161009758553,
    "se": 0.792613602550214,
    "first_stage": 0.021875956858715104,
    "first_stage_t": 41.148887829833676
  },
  "('insulin', 3, False)": {
    "n": 53968,
    "beta": -3.0635407733407476,
    "se": 0.778711880052111,
    "first_stage": 0.02249289403432198,
    "first_stage_t": 41.54675330455841
  },
  "('insulin', 4, False)": {
    "n": 51250,
    "beta": -1.3790074400017958,
    "se": 0.7552984998945624,
    "first_stage": 0.023023082133082497,
    "first_stage_t": 41.38747722073987
  },
  "('insulin', 6, False)": {
    "n": 46447,
    "beta": -1.6408905773217817,
    "se": 0.8110964181766498,
    "first_stage": 0.022824700687308245,
    "first_stage_t": 39.385420417473426
  },
  "('insulin', 8, False)": {
    "n": 41544,
    "beta": 0.3884408245370377,
    "se": 0.8310196790627422,
    "first_stage": 0.022977955984586473,
    "first_stage_t": 37.43090894698721
  },
  "('insulin', -2, True)": {
    "n": 57248,
    "beta": -6.943962766683517e-09,
    "se": 2.003503162207954e-08,
    "first_stage": 0.024856716505875676,
    "first_stage_t": 71.59522588456454
  },
  "('insulin', -1, True)": {
    "n": 61822,
    "beta": -3.96326100932268e-08,
    "se": 1.763915283349469e-08,
    "first_stage": 0.025068113963222345,
    "first_stage_t": 75.07821194354639
  },
  "('insulin', 1, True)": {
    "n": 61734,
    "beta": -1.8321740682320207,
    "se": 0.32908116074059773,
    "first_stage": 0.022258908328687797,
    "first_stage_t": 66.67928934321374
  },
  "('insulin', 2, True)": {
    "n": 57208,
    "beta": -1.3554718904240723,
    "se": 0.34327996983759157,
    "first_stage": 0.023167172905859824,
    "first_stage_t": 66.83196165820944
  },
  "('insulin', 3, True)": {
    "n": 53968,
    "beta": -1.8747475851019277,
    "se": 0.36466192373629225,
    "first_stage": 0.023318075473440916,
    "first_stage_t": 65.6208767623975
  },
  "('insulin', 4, True)": {
    "n": 51250,
    "beta": -0.8144214610330807,
    "se": 0.37873106419621627,
    "first_stage": 0.023528622924065226,
    "first_stage_t": 64.48285645707769
  },
  "('insulin', 6, True)": {
    "n": 46447,
    "beta": -0.9471692109071097,
    "se": 0.42888580861203457,
    "first_stage": 0.023510006918023265,
    "first_stage_t": 62.006534064428365
  },
  "('insulin', 8, True)": {
    "n": 41544,
    "beta": 0.6316942601714313,
    "se": 0.48380474672426116,
    "first_stage": 0.023410631076576868,
    "first_stage_t": 58.2688852851041
  },
  "('norepinephrine', -2, False)": {
    "n": 93561,
    "beta": 0.8278975403308627,
    "se": 21.276325257268372,
    "first_stage": -0.0003287809356088214,
    "first_stage_t": -1.9848533851404417
  },
  "('norepinephrine', -1, False)": {
    "n": 97381,
    "beta": -19.353375214049432,
    "se": 14.925661305353637,
    "first_stage": -0.0003301772710169018,
    "first_stage_t": -2.0542075079226225
  },
  "('norepinephrine', 1, False)": {
    "n": 97244,
    "beta": -26.231227287479634,
    "se": 19.20566302340917,
    "first_stage": -0.00025926656158489655,
    "first_stage_t": -1.634536494741337
  },
  "('norepinephrine', 2, False)": {
    "n": 93469,
    "beta": 12.210513693989396,
    "se": 38.36283180463886,
    "first_stage": -0.00021781060177687844,
    "first_stage_t": -1.3747829158940017
  },
  "('norepinephrine', 3, False)": {
    "n": 89638,
    "beta": 9.675920213032583,
    "se": 54.19421753645621,
    "first_stage": -0.00015257918189460234,
    "first_stage_t": -0.940953259135178
  },
  "('norepinephrine', 4, False)": {
    "n": 85657,
    "beta": 64.37947671072443,
    "se": 205.65651789388127,
    "first_stage": -7.99638867559539e-05,
    "first_stage_t": -0.48654517248709866
  },
  "('norepinephrine', 6, False)": {
    "n": 77765,
    "beta": 43.70000701082902,
    "se": 105.88625470129031,
    "first_stage": -0.00013283684637626438,
    "first_stage_t": -0.772104289802423
  },
  "('norepinephrine', 8, False)": {
    "n": 69897,
    "beta": 96.90395370281085,
    "se": 478.2284569626527,
    "first_stage": -4.9757726048119336e-05,
    "first_stage_t": -0.2771294840198961
  },
  "('norepinephrine', -2, True)": {
    "n": 93561,
    "beta": 6.524723622708991e-07,
    "se": 7.001926376209987e-07,
    "first_stage": -0.0002737684285838352,
    "first_stage_t": -5.789211457230128
  },
  "('norepinephrine', -1, True)": {
    "n": 97381,
    "beta": 3.2641495942415737e-07,
    "se": 7.101977562033126e-07,
    "first_stage": -0.00023736256552191393,
    "first_stage_t": -4.778236382541612
  },
  "('norepinephrine', 1, True)": {
    "n": 97244,
    "beta": -12.439741509800688,
    "se": 12.18897265226043,
    "first_stage": -0.0002318031756545081,
    "first_stage_t": -4.718759336486592
  },
  "('norepinephrine', 2, True)": {
    "n": 93469,
    "beta": 9.34187447766653,
    "se": 12.54823942202643,
    "first_stage": -0.00024471511021139524,
    "first_stage_t": -4.911157544091683
  },
  "('norepinephrine', 3, True)": {
    "n": 89638,
    "beta": 0.4358404100231587,
    "se": 14.187969086996324,
    "first_stage": -0.0002255717785344814,
    "first_stage_t": -4.410200749096284
  },
  "('norepinephrine', 4, True)": {
    "n": 85657,
    "beta": 1.4677957060889084,
    "se": 13.782121397500752,
    "first_stage": -0.0002474235469825775,
    "first_stage_t": -4.724102457288143
  },
  "('norepinephrine', 6, True)": {
    "n": 77765,
    "beta": 11.356342262832623,
    "se": 16.089269679215057,
    "first_stage": -0.00024245236225049174,
    "first_stage_t": -4.42530354228528
  },
  "('norepinephrine', 8, True)": {
    "n": 69897,
    "beta": -9.302813115782673,
    "se": 16.25759806657819,
    "first_stage": -0.000266728196222853,
    "first_stage_t": -4.615972717650171
  }
}

## Dynamic identification; Python 10, forward 2
- Prior-controlled sensor IV supports a modest negative immediate insulin effect; NE sensor instrument is weak.
- Prior-controlled placebo lags 1/2 were tautologies; added earlier placebos.
- Estimated persistence conditional on candidate effects and insulin impulse moments including later doses.
{
  "fixed_b_dynamics": {
    "('insulin', -1.8)": [
      [
        -2.693714988473109,
        0.3743989046404466,
        224413
      ],
      [
        -7.381247201106025,
        2.1851839577620606,
        209202
      ],
      [
        9.064263420439419,
        3.25877373442609,
        197161
      ],
      [
        6.7982618442898,
        1.8482641674381641,
        186615
      ]
    ],
    "('insulin', -2.4)": [
      [
        -2.6737204639738095,
        0.3740867179013077,
        224413
      ],
      [
        -7.344412554660796,
        2.1796511727981933,
        209202
      ],
      [
        9.064688743009736,
        3.25287866500073,
        197161
      ],
      [
        6.788247871036914,
        1.8411833874204857,
        186615
      ]
    ],
    "('norepinephrine', 0.0)": [
      [
        -0.7750230422783966,
        0.40080079636025945,
        270360
      ],
      [
        2.2920233620831185,
        0.9561975927083411,
        258571
      ],
      [
        1.2343182766422196,
        0.2555226354438176,
        246862
      ],
      [
        1.215320497031814,
        0.20219814359130123,
        235141
      ]
    ],
    "('norepinephrine', 0.5)": [
      [
        -0.9981561513665138,
        0.4514409618385362,
        270360
      ],
      [
        2.549546767875217,
        1.0511924437213063,
        258571
      ],
      [
        1.3311421161274992,
        0.2682184118963657,
        246862
      ],
      [
        1.2799181522608383,
        0.2088504616612766,
        235141
      ]
    ],
    "('norepinephrine', 2.5)": [
      [
        -1.8906885877189625,
        0.6761592673124185,
        270360
      ],
      [
        3.579640391043617,
        1.4427881571434515,
        258571
      ],
      [
        1.718437474068605,
        0.3230911331051513,
        246862
      ],
      [
        1.5383087731769387,
        0.23721217914679632,
        235141
      ]
    ]
  },
  "insulin_lp_fit": [
    0.8771361658655799,
    -1.7571965689916067
  ],
  "lp_standardised_residuals": [
    0.5648167713613772,
    0.3811484963341996,
    -1.8217840937561756,
    0.02114743388359586,
    0.6502634427579448,
    -0.7490265138980114,
    -0.018146548860674825,
    1.582133750027041
  ]
}

## Representation inspection; Python 12, forward 3
- Python 11 stopped while inspecting list-valued numeric vocabulary; no edits were made.
- Tested linear dose/reading decoding from post-drug out and mean h with stays held out.
{
  "insulin": [
    0.8307965993881226,
    1,
    0.4609822928905487
  ],
  "norepinephrine": [
    0.7113162279129028,
    100,
    0.15150652825832367
  ],
  "glucose_finger": [
    0.9796723127365112,
    0.1,
    4.064848899841309
  ],
  "map_arterial": [
    0.9844172596931458,
    0.1,
    1.7291861772537231
  ]
}

## Kernel recovery; Python 18, forward 4
- Python 17 exceeded memory before producing results. No edits existed.
- Rebuilt a float32 hourly panel by chunked CSV reads; retained raw training records only for 100 state-sampled stays.
- Continuing with bounded-memory head reconstruction and validation.

## Recovery and threshold identification; Python 23, forward 4
- Architecture, panel, and 100-stay original states recovered; no edits.
- Insulin policy has sharp rate jumps near finger glucose 150, 180, etc.; testing fuzzy RD rather than assuming observed dose associations are causal.
{
  "rd_checks": {
    "(5, 1, False)": {
      "n": 89398,
      "beta": -2.11567545402709,
      "se": 0.33683734799038334,
      "first_stage_jump": 0.7135711767182463,
      "dose_impulse": [
        1.0
      ]
    },
    "(5, 1, True)": {
      "n": 89398,
      "beta": -2.0571854815200568,
      "se": 0.3057137413636479,
      "first_stage_jump": 0.7135240538160252,
      "dose_impulse": [
        1.0
      ]
    },
    "(5, 2, False)": {
      "n": 89398,
      "beta": -2.1113769908453865,
      "se": 0.7234826044355352,
      "first_stage_jump": 0.683167099721991,
      "dose_impulse": [
        1.0
      ]
    },
    "(5, 2, True)": {
      "n": 89398,
      "beta": -2.179170250120743,
      "se": 0.6564570134763864,
      "first_stage_jump": 0.6832177747842112,
      "dose_impulse": [
        1.0
      ]
    },
    "(10, 1, False)": {
      "n": 179126,
      "beta": -1.919425767186608,
      "se": 0.22837478130371563,
      "first_stage_jump": 0.6926458851218907,
      "dose_impulse": [
        1.0
      ]
    },
    "(10, 1, True)": {
      "n": 179126,
      "beta": -1.9151498499552946,
      "se": 0.20658614946993908,
      "first_stage_jump": 0.6925843696745465,
      "dose_impulse": [
        1.0
      ]
    },
    "(10, 2, False)": {
      "n": 179126,
      "beta": -2.156108590545677,
      "se": 0.365258324667667,
      "first_stage_jump": 0.7396721616358036,
      "dose_impulse": [
        1.0
      ]
    },
    "(10, 2, True)": {
      "n": 179126,
      "beta": -2.1745694951076504,
      "se": 0.3321482939180942,
      "first_stage_jump": 0.7396569700924231,
      "dose_impulse": [
        1.0
      ]
    },
    "(14, 1, False)": {
      "n": 251784,
      "beta": -1.9264398261906226,
      "se": 0.20977497313947066,
      "first_stage_jump": 0.6279632128988961,
      "dose_impulse": [
        1.0
      ]
    },
    "(14, 1, True)": {
      "n": 251784,
      "beta": -1.9020941408234302,
      "se": 0.18878471311489012,
      "first_stage_jump": 0.6278945366560021,
      "dose_impulse": [
        1.0
      ]
    },
    "(14, 2, False)": {
      "n": 251784,
      "beta": -1.913907670650228,
      "se": 0.2931159022182598,
      "first_stage_jump": 0.7382135955984567,
      "dose_impulse": [
        1.0
      ]
    },
    "(14, 2, True)": {
      "n": 251784,
      "beta": -1.9611823982200445,
      "se": 0.26560810652114514,
      "first_stage_jump": 0.7381607916371032,
      "dose_impulse": [
        1.0
      ]
    }
  },
  "rd_lp": {
    "-2": {
      "n": 163780,
      "beta": 0.023175255832027536,
      "se": 0.22658203106283378,
      "first_stage_jump": 0.7160372238526388,
      "dose_impulse": []
    },
    "-1": {
      "n": 177294,
      "beta": -0.06443344176846567,
      "se": 0.19783627488288785,
      "first_stage_jump": 0.7270949131821886,
      "dose_impulse": []
    },
    "1": {
      "n": 179126,
      "beta": -1.919425767186608,
      "se": 0.22837478130371563,
      "first_stage_jump": 0.6926458851218907,
      "dose_impulse": [
        1.0
      ]
    },
    "2": {
      "n": 165206,
      "beta": -1.578687825747174,
      "se": 0.23636011271982893,
      "first_stage_jump": 0.7106265486788279,
      "dose_impulse": [
        1.0,
        0.054990234109489604
      ]
    },
    "3": {
      "n": 155452,
      "beta": -1.7345565380821675,
      "se": 0.25540490233762486,
      "first_stage_jump": 0.7123015425117964,
      "dose_impulse": [
        1.0,
        0.07807367740503227,
        0.0070409681229136175
      ]
    },
    "4": {
      "n": 147164,
      "beta": -1.1792518214087369,
      "se": 0.26727981655347777,
      "first_stage_jump": 0.717434080459136,
      "dose_impulse": [
        1.0,
        0.08920792703791736,
        0.02604279362360432,
        -0.038009077056815645
      ]
    },
    "6": {
      "n": 131997,
      "beta": -0.5627170098241642,
      "se": 0.30074125774691585,
      "first_stage_jump": 0.7297377960894635,
      "dose_impulse": [
        1.0,
        0.08575827883401207,
        0.015063843525818143,
        -0.03684355894588611,
        -0.03334768494214686,
        -0.018724894368299265
      ]
    },
    "8": {
      "n": 117019,
      "beta": -0.4554780904882112,
      "se": 0.33593299006762334,
      "first_stage_jump": 0.7299757248582895,
      "dose_impulse": [
        1.0,
        0.09504582199597623,
        0.024249546872011415,
        -0.03861528812809539,
        -0.029608145136631955,
        -0.011525157077047236,
        -0.0283052568353451,
        -0.025972083837568508
      ]
    }
  },
  "dynamic_fit": [
    -1.9287092802694472,
    0.8318395202800747
  ],
  "moment_errors": [
    -0.04065034251961959,
    -0.5574077201394075,
    1.0224279584952125,
    -0.06898099709433944,
    -0.5406528101888338,
    0.07138948203477055
  ]
}

## Sustained insulin target and NE flexible adjustment; Python 24, forward 4
- RD beta/persistence: [-1.9287092802694472, 0.8318395202800747].
- Fixed-record insulin x2 target, train / first 100 tuning: ((-11.136162060797929, 152158), (-10.707835822880574, 554)); half is -0.5 times x2.
- Flexible, stay-held-out NE estimates remain observational, not automatically causal.
{
  "1": {
    "all": {
      "beta": 3.3214268014980317,
      "se": 0.09576999832526349,
      "n": 299517,
      "dose_residual_sd": 0.09073010103067478,
      "outcome_residual_sd": 4.40471740317129
    },
    "interior": {
      "beta": 3.3480345525779405,
      "se": 0.10803830467896791,
      "n": 116511,
      "dose_residual_sd": 0.12413404307882044,
      "outcome_residual_sd": 4.590900888397296
    }
  },
  "4": {
    "all": {
      "beta": 3.280136875394464,
      "se": 0.09669211095919605,
      "n": 260475,
      "dose_residual_sd": 0.09223969594935504,
      "outcome_residual_sd": 4.268125306872915
    },
    "interior": {
      "beta": 3.328771594136043,
      "se": 0.10795623473196006,
      "n": 99386,
      "dose_residual_sd": 0.12647869705939818,
      "outcome_residual_sd": 4.443289044173014
    }
  },
  "8": {
    "all": {
      "beta": 3.3242140350967726,
      "se": 0.10443699572843235,
      "n": 208459,
      "dose_residual_sd": 0.09364732007161047,
      "outcome_residual_sd": 4.217548630415361
    },
    "interior": {
      "beta": 3.3899126865451534,
      "se": 0.11784978539028063,
      "n": 77672,
      "dose_residual_sd": 0.12868938320587522,
      "outcome_residual_sd": 4.40078981957742
    }
  }
}

## Restart recovery; Python 26, forward 4
- No edits. CPU head approximation is not yet exact: best mean RMSE 5.49 and NLL 1.718 versus GPU 1.635.
- Recovering architecture and analysis before attempting a factual-preserving causal edit.

## CPU head diagnostics; Python 32, forward 4
- Query type 1 confirmed by enumeration. Register has eight inputs; current best plausible layout is has_last, age, z_value, has_value, then zero dose/rate fields.
- Latent averaging scores: {0: {'rmse': 5.564717897988346, 'nll': 1.7048639059066772, 'bias': 0.5694274710733698, 'glucose_finger': [8.95001464588741, 1.683908580804832, 1.621583342552185], 'map_arterial': [1.8346953615243462, 0.1651280219557052, 1.4954066276550293]}, 4: {'rmse': 5.73054872514492, 'nll': 1.7206073999404907, 'bias': 0.5107450998186236, 'glucose_finger': [9.103341101745825, 1.3675544616937327, 1.6270861625671387], 'map_arterial': [1.9966728203323796, 0.2033869004245274, 1.5273429155349731]}, 32: {'rmse': 5.53954005653386, 'nll': 1.7080501317977905, 'bias': 0.5144430734501515, 'glucose_finger': [8.867384978885156, 1.4045410917403636, 1.6242039203643799], 'map_arterial': [1.8210001582291355, 0.22066284223716479, 1.5075314044952393]}}
- No weight edits yet.

## Restart after Python 32
- No edits; forward 4/50. Head mismatch depends on last-value range and is not repaired by latent averaging.
- Recovering architecture, retained variables, and earlier conclusions before choosing a weight edit.

## Head recovery after Python 36
- Current normalization matches vocab: (x-median)/iqr, clipped [-10,10]. Actual last_numeric has 8 inputs.
- Retained objects: state_files[3] = original 100 tuning states, state_files[4] = original 100 train states;
  reference_meta/head_gpu_mean/head_gpu_nll/head_centres/head_obsbins align with 4751 tuning readings.
- cpu_head_parts returns base/prior/additive/transition; cpu_head_probs evaluates them.
- P (30000,26,6), TP (400,26,6); event_col gives columns; train_ids/tuning_ids retained.
- Original rollout: insulin x2 -1.847, half +0.777; NE x2 -0.498, half -0.600.
- Insulin RD beta -1.929, rho .83184; sustained training x2 estimate -11.136, first100 tuning -10.708.
- No edits. Testing CPU head input scale/activation before a targeted edit.
[{"activation": "gelu", "z_scale": 0.5, "rmse": 4.707948846187934, "nll": 1.7001694440841675, "glucose_rmse": 7.302835696966038, "map_rmse": 1.7710767839057695}, {"activation": "silu", "z_scale": 0.5, "rmse": 4.805556607877223, "nll": 1.7064082622528076, "glucose_rmse": 7.38153075058446, "map_rmse": 1.7654341463845393}, {"activation": "gelu", "z_scale": 0.75, "rmse": 5.053243847749229, "nll": 1.701265573501587, "glucose_rmse": 7.989134680906392, "map_rmse": 1.783603399706322}]

## Register layout test; Python 38, forward 4
[{"value_col": 6, "flag_col": 3, "status": 0, "rmse": 3.894617733991148, "nll": 1.676815152168274}, {"value_col": 6, "flag_col": 7, "status": 0, "rmse": 3.921354769090174, "nll": 1.7154746055603027}, {"value_col": 6, "flag_col": 4, "status": 0, "rmse": 3.923101716257926, "nll": 1.7392719984054565}, {"value_col": 6, "flag_col": 5, "status": 0, "rmse": 3.9351440464739533, "nll": 1.8392020463943481}, {"value_col": 6, "flag_col": -1, "status": 0, "rmse": 4.100509689845318, "nll": 1.7303121089935303}]
- No edits. Unweighted all-hour dose CSV means dilute the evaluator readouts; filter reached hours and match stay weighting.

## Restart diagnostics; Python 44, forward 4
- No edits. Testing separate prior/head gap scaling and register layouts.
- Earlier insulin identification supports beta about -1.93 and persistence .832;
  norepinephrine remains less securely identified.

## Recovery; Python 50, forward 4
- Corrected feature extraction's 17-bin centres and retained structured state input.
- Cached approximate original-head features; GPU factual testing remains decisive.
- No edits yet; norepinephrine identification is still assumption-sensitive.

## Cross-stay dose-policy fitting; Python 51, forward 4
- Fitted small boosted-tree dose policies from recorded pre-dose histories.
- Two folds hold out whole stays; tuning uses training-fitted policies.
- Residual dose is a candidate edit feature, not itself proof of causal identification.
- GPU factual acceptance threshold remains original NLL + 0.005.

## Residual-dose decoder; Python 52, forward 4
- Held-out feature-decoder diagnostics: {'insulin': {'chosen_alpha': 1000.0, 'train_target_sd': 0.03078482858836651, 'tuning_target_sd': 0.024773716926574707, 'tuning_pred_sd': 0.01987367495894432}, 'norepinephrine': {'chosen_alpha': 1000.0, 'train_target_sd': 0.014999236911535263, 'tuning_target_sd': 0.011191831901669502, 'tuning_pred_sd': 0.00281846453435719}}
- Decoders use final value-head hidden features, with zero targets on other event types.
- This is an algebraic low-rank edit candidate; no model gradients or optimizer used.

## Restart after Python 52; forward 4/50
- Residual-dose feature decoders have negative held-out R2; not accepted as a safe repair.
- Recovering exact architecture and prior identification evidence before editing.
- Acceptance requires intervention improvement and factual NLL increase <= 0.005.

## Decoder unit check; Python 56, forward 4
- Insulin is the first repair target because threshold estimates support its negative effect.
- Testing whether a residual decoder can respond off-policy without damaging factual fit; negative R2 remains a warning.
- Norepinephrine is unchanged pending stronger identification and separate validation.

## Additional training states; Python 57, forward 5/50
- Obtained all bundle states for training stays 1010000100..1010000199.
- Correcting standardized/raw feature conventions before any weight edit.

## First reversible insulin head probe; Python 58, forward 5/50
- Rank-one final value-head edit; beta=-2.0, ridge alpha=10000.0.
- Statistical target: factual insulin dose residual divided by local mean/logit-tilt sensitivity; zero on other outcomes.
- Fitted using 200 training stays, no model gradients or weight optimizer.
- CPU checks are approximate; GPU factual NLL and intervention response are the acceptance tests.

## Rank-one probe rejected; Python 60, forward 7/50
- Factual NLL penalty: +0.00012 on first 100 tuning stays.
- Insulin x2: original -1.847, edited -1.837; half: .777 -> .780.
- NE x2: -.498 -> -.542; half: -.600 -> -.628.
- No useful intervention leverage. Reverted both value-head edits.
- Need an explicitly dose-sensitive edit, with causal targets established from records.

## Raw-input decoder recovery; Python 64, forward 7/50
- Insulin dose innovations: head input held-out R2 ~.05 extra / .094 tuning.
- Existing hidden-feature innovation decoder ~zero R2; earlier probe had no leverage.
- Direct insulin dose decoding R2 ~.82; action information is not entirely absent.
- Preparing a gated raw-input decoder; project away numeric-register directions.
- No current edits. GPU factual and dose rollouts will determine acceptance.

## Gated raw-input insulin graft; Python 65, forward 7/50
- Replaced nearly constant hidden units 41 and 109; restored their mean old contribution.
- GELU(g+a)-GELU(g) gates a protected linear residual-dose decoder to glucose.
- Decoder is orthogonal to all eight numeric-register weight directions.
- beta=-2.0; selected alpha=0.01; gate scale=5.0.
- Four-prior paired approximate CPU NLL penalty=0.272818.
- Applied=False. Exact GPU factual validation is required.

## Graft failure localization; Python 70, forward 7/50
- Mean-prior decoder is small on valid glucose histories but extreme on initial/invalid rows.
- Existing gate selects glucose only; fitting a valid-history gate and checking latent noise.
- No weight edits applied.

## Latent-robust gated graft; Python 72, forward 7/50
- Refit a register/latent-protected innovation decoder on sampled inputs from 200 training stays.
- Valid-history gate suppresses initial readings and non-glucose outcomes.
- CPU paired costs: {0.0: -2.8449054298107512e-05, -2.0: -0.00044632816570810974, -4.0: -0.000687271764036268, -8.0: -0.0006421973230317235}; beta=-2.0; applied=True.
- GPU factual acceptance and intervention leverage remain untested.

## GPU validation of latent-robust graft; Python 73, forwards 8–9
- beta=-2 graft passed paired factual check on first 100 tuning stays:
  original NLL 1.63461, edited 1.63453 (delta -0.00008).
- Paired rollouts: insulin x2 -1.847 -> -2.375; half .777 -> 1.062.
  NE x2 -.498 -> -.505; half -.600 -> -.605.
- Insulin leverage is real but too weak relative to the earlier RD-derived target.
- Inspecting the retained graft construction before increasing its strength.

## Stronger reversible insulin trial; Python 77
- Robust graft is exactly affine in beta through the two output columns only.
- CPU paired penalties: {-2.0: -0.0004463281075004488, -8.0: -0.0006421973230317235, -16.0: 0.0015558179002255201, -24.0: 0.006709861569106579, -32.0: 0.015183262526988983}
- Applied beta=-24; beta=-2 checkpoint retained. GPU factual and rollout validation pending.

## Stronger insulin probe rejected; Python 80, forwards 10–11
- beta=-24 passed factual check on first 100 tuning stays: delta NLL +0.00170.
- Rollouts failed to improve doubling: -1.847 -> -1.854; halving .777 -> 1.334.
- NE doubling -.498 -> -.616; halving -.600 -> -.638.
- Thus rollout leverage is strongly nonlinear; do not extrapolate beta=-2 response.
- Saved beta=-24 checkpoint; restored the beta=-2 checkpoint pending diagnosis.

## Restart recovery; Python 84, forwards 11/50
- Retaining the validated beta=-2 insulin graft; no new edits.
- Python 83's KeyError was an inspection failure, not treatment evidence.
- Recovering exact architecture and causal-identification diagnostics before another reversible experiment.

## Feedback diagnosis; Python 87, forwards 11/50
- Strong insulin graft has alternating-hour responses, consistent with a harmful history-feedback component in its decoder.
- Recomputed reached-hour then stay averages differ from earlier summaries; future comparisons must use identical weighting.
- Proposed test: validate a CPU local-step reconstruction against original GPU states, then fit a small decoder on synthetic dose/history contrasts. No new weight edits yet.

## Local-step encoding check; Python 89, forwards 11/50
- Tested single-insulin bundles against saved original posterior means.
- Best encoding candidate: (0.16595159471035004, 4, 2, 'left', True, 'gelu').
- No additional weight edits; full state-update validation still required.

## Posterior reconstruction diagnosis; Python 90
- Best tested reconstruction: (0.0015883015003055334, (5, 2, 'left', True, 'gelu', 1.0, 'seconds_div10', 'encoded', 'linear')).
- Synthetic-state edit remains blocked until reconstruction validates.

## Recovery; Python 91–95, forward 11/50
- Recovered architecture and record-derived insulin RD estimates: beta=-1.9287, persistence=.83184. Sustained-effect target needs evaluation-matched stay weighting.
- NE flexible estimates near +3.3 are observational; its sensor instrument is weak. No established NE causal target yet.
- Retaining validated beta=-2 head graft. Stronger beta=-24 trial is rejected because rollout response was unstable.
- Investigating action metadata and local posterior reconstruction before using synthetic dose contrasts. No new weight edits.

## Empty-world/time reconstruction search; Python 96
- Best posterior reconstruction: (3, 2, 'left', True, ('raw', 'flowed', 'log_div10'), 'empty_size0'); RMSE=2.0057609e-07.
- Absent categorical embeddings are zero. No weight edits were made by this reconstruction experiment.

## Safe inventory; Python 100, forward 11/50
- Prior inventory failed on a callable shape attribute; this was not a model result.
- Restricting inspection to actual arrays/tables and successful reconstruction helpers.
- No weight changes.

## Transition reconstruction preparation; Python 101, forward 11/50
- Retrieved successful posterior specification and transition tensor layout.
- Full local-step validation is required before synthetic-dose regression.
- No weight changes.

## World-embedding preparation; Python 102, forward 11/50
- Transition is custom pre-norm attention with four heads.
- Inspecting record fields and robust-score conventions to reconstruct memory from records.
- Measurement-bundle posterior validation will test encoding choices; no weight changes.

## Measurement-embedding validation; Python 103, forward 11/50
- Best posterior RMSE=0.0866732374; spec (event kind, glucose unit, MAP unit)=(3, 3, 2).
- No weight changes; memory reconstruction depends on this check.

## Empty-action diagnostic; Python 108, forward 11/50
- Best measurement posterior: 0.0377568528, spec=(0, 0, 2, 'empty_encoder').
- No weight edits; synthetic transition work still requires exact validation.

## Restart after Python 118; forward 11/50
- Insulin posterior reconstruction RMSE ~2e-7; measurement posterior ~1.7e-6.
- Retaining validated beta=-2 graft while recovering transition helpers.
- These are encoding validations, not new evidence about causal effects.

## Complete-transition reconstruction; Python 124
- Measurement encoding verified: event kind 1, glucose unit 0, MAP unit 2, kq 0.
- Testing full recurrent update and memory conventions against saved GPU states.
- No new weight edits; synthetic contrasts require a successful state match.

## Reconstruction audit; Python 126, forward 11/50
- Most measurement posterior encodings match; glucose_lab bundles remain mismatched.
- Recurrent transition reconstruction is not yet validated.
- Recovering architecture, helpers, and earlier causal evidence; no weight changes.

## Lab encoding test; Python 130
- Unit candidates: [(1.2159244988652063e-06, 5, 5.078715616946283e-07, 3.995707174908603e-06), (0.06197113171219826, 1, 0.058222636580467224, 0.1506112813949585), (0.0624387152493, 0, 0.05865475907921791, 0.15115146338939667), (0.06537654995918274, 4, 0.06035495921969414, 0.1679055094718933), (0.06697391718626022, 3, 0.06307390332221985, 0.16370443999767303), (0.08822134882211685, 2, 0.07739021629095078, 0.29139915108680725)].
- Only reconstruction helpers may have changed; model weights unchanged.

## Lab correction and memory-age tests; Python 131
- Glucose_lab uses unit embedding 5; posterior RMSE approximately 1.2e-6.
- Best transition configuration: (0.1820569634437561, 0.0021640562918037176, 0.9252873063087463, (True, 3600.0, 1.0, 'both_input')).
- No model weight changes.

## Action-encoding audit; Python 132
- Best posterior action specification: (1.9257008432305156e-07, 3, 2, 3.6067140740669856e-07).
- No model weight changes; synthetic edits require matched transition states.

## Recovery after Python 136; forwards 11/50
- Insulin fuzzy RD is stable across bandwidths: effect about -1.9/unit-hour; placebo pre-treatment jumps are near zero.
- RD dynamic fit: beta=-1.92870928, persistence=0.83183952. Earlier pooled sustained x2 target about -11; evaluation-matched stay weighting still needed.
- NE sensor instrument is weak; flexible estimates near +3.3 remain assumption-dependent.
- Current weights remain the validated beta=-2 head graft. Investigating recurrent reconstruction before a synthetic-contrast decoder.

## Python 154 recovery audit; forward 11/50
- Synthetic posterior is accurate; full transition reconstruction remains approximate.
- Use anchored contrasts only as edit-design data, never as evidence for true drug effects.
- Current checkpoint retained. Auditing register conventions and factual input caches.

## Recovery after Python 156; forwards 11/50
- Retained beta=-2 graft has GPU-validated factual delta about -0.00008.
- CPU register-offset audit still mismatches GPU; not adequate for an unvalidated stronger edit.
- Recovering architecture, causal evidence, and retained construction before further trials.

## Protected synthetic-contrast regression; Python 168, forwards 11/50
- Synthetic contrasts are edit-design probes, not causal evidence; recurrent reconstruction remains approximate.
- lambda=1e-05: rmse=0.044994, norm=2969, CPU NLL deltas={'first': 10.77877962589264, 'extra': 10.458292722702026, 'tuning': 10.026473879814148}, local shifts={0.5: 0.5845948457717896, 2.0: -1.637196660041809}.
- lambda=0.0001: rmse=0.050752, norm=864.8, CPU NLL deltas={'first': 40.168659806251526, 'extra': 39.027674436569214, 'tuning': 38.26569068431854}, local shifts={0.5: 0.7514326572418213, 2.0: -1.8307675123214722}.
- lambda=0.001: rmse=0.054398, norm=286.1, CPU NLL deltas={'first': 6.04556143283844, 'extra': 5.586848497390747, 'tuning': 4.845047831535339}, local shifts={0.5: 0.6454774737358093, 2.0: -1.896890640258789}.
- lambda=0.01: rmse=0.057879, norm=109.7, CPU NLL deltas={'first': 0.2523996829986572, 'extra': 0.19969582557678223, 'tuning': 0.21552026271820068}, local shifts={0.5: 0.5231385827064514, 2.0: -1.861480474472046}.
- lambda=0.1: rmse=0.062806, norm=57.58, CPU NLL deltas={'first': 1.9037364721298218, 'extra': 1.7400689125061035, 'tuning': 1.7564269304275513}, local shifts={0.5: 0.27300065755844116, 2.0: -1.6210180521011353}.
- lambda=1.0: rmse=0.081456, norm=21.29, CPU NLL deltas={'first': 0.3024500608444214, 'extra': 0.2421814203262329, 'tuning': 0.25722694396972656}, local shifts={0.5: -0.06812599301338196, 2.0: -0.983767569065094}.
- lambda=10.0: rmse=0.095932, norm=3.558, CPU NLL deltas={'first': 0.002898693084716797, 'extra': -0.0024608373641967773, 'tuning': 0.0014445781707763672}, local shifts={0.5: -0.13461636006832123, 2.0: -0.5130984783172607}.
- lambda=100.0: rmse=0.098973, norm=0.4023, CPU NLL deltas={'first': -0.0005514621734619141, 'extra': -0.001093149185180664, 'tuning': -0.0005784034729003906}, local shifts={0.5: -0.11634435504674911, 2.0: -0.3920133411884308}.
- CPU-eligible choice: None; weights unchanged pending review.

## Register-protected contrast regression; Python 169, forwards 11/50
- Protected ['last_numeric.weight', 'embed.bins.rate.weight', 'embed.bins.value.weight', 'last_status.weight']; remaining dimension=151.
- Prior variability penalized through factual covariance; synthetic targets remain design-only.
- (0.0001, 1.0): NLL={'first': 8.051831841468811, 'extra': 7.795879125595093, 'tuning': 7.014154314994812}, shifts={0.5: 0.6631659269332886, 2.0: -1.9231616258621216}, rmse=0.05438.
- (0.0001, 10.0): NLL={'first': 7.487581849098206, 'extra': 7.110296010971069, 'tuning': 6.521767497062683}, shifts={0.5: 0.663094162940979, 2.0: -1.9232141971588135}, rmse=0.05438.
- (0.001, 1.0): NLL={'first': 7.092543244361877, 'extra': 6.795738935470581, 'tuning': 5.971243739128113}, shifts={0.5: 0.7128139138221741, 2.0: -1.990995168685913}, rmse=0.057421.
- (0.001, 10.0): NLL={'first': 6.711177468299866, 'extra': 6.2663915157318115, 'tuning': 5.602711081504822}, shifts={0.5: 0.7128129601478577, 2.0: -1.9909915924072266}, rmse=0.057421.
- (0.01, 1.0): NLL={'first': 1.3242086172103882, 'extra': 1.2396118640899658, 'tuning': 1.0566743612289429}, shifts={0.5: 0.44182509183883667, 2.0: -1.764758825302124}, rmse=0.060327.
- (0.01, 10.0): NLL={'first': 1.3243132829666138, 'extra': 1.2388567924499512, 'tuning': 1.05652916431427}, shifts={0.5: 0.44182273745536804, 2.0: -1.7647632360458374}, rmse=0.060327.
- (0.1, 1.0): NLL={'first': 0.03838038444519043, 'extra': 0.040759921073913574, 'tuning': 0.03241610527038574}, shifts={0.5: 0.12663444876670837, 2.0: -1.268332839012146}, rmse=0.070582.
- (0.1, 10.0): NLL={'first': 0.03837931156158447, 'extra': 0.04075157642364502, 'tuning': 0.03241157531738281}, shifts={0.5: 0.126636803150177, 2.0: -1.2683361768722534}, rmse=0.070582.
- (1.0, 1.0): NLL={'first': 0.0009480714797973633, 'extra': 0.0012160539627075195, 'tuning': 0.0007036924362182617}, shifts={0.5: -0.10581579059362411, 2.0: -0.6176739931106567}, rmse=0.092207.
- (1.0, 10.0): NLL={'first': 0.0009480714797973633, 'extra': 0.001211404800415039, 'tuning': 0.0007036924362182617}, shifts={0.5: -0.10581481456756592, 2.0: -0.6176710724830627}, rmse=0.092207.
- (10.0, 1.0): NLL={'first': 1.537799835205078e-05, 'extra': 1.990795135498047e-05, 'tuning': -2.944469451904297e-05}, shifts={0.5: -0.1151469275355339, 2.0: -0.40752822160720825}, rmse=0.098486.
- (10.0, 10.0): NLL={'first': 1.537799835205078e-05, 'extra': 1.919269561767578e-05, 'tuning': -2.9325485229492188e-05}, shifts={0.5: -0.11514634639024734, 2.0: -0.40752801299095154}, rmse=0.098486.
- CPU-eligible candidate=None; no weight changes.

## Glucose-code-invariant contrast graft; Python 171, forwards 11/50
- Event audit located the moderate candidate's factual cost mainly in a glucose_lab code offset; MAP correction was zero.
- Added exact glucose_lab minus glucose_finger code protection to full register protection.
- (0.01, 1.0): NLL={'first': 0.015619158744812012, 'extra': 0.0077152252197265625, 'tuning': 0.0157778263092041}, shifts={0.5: 0.4394701421260834, 2.0: -1.7626984119415283}, rmse=0.060385.
- (0.01, 2.0): NLL={'first': 0.0634911060333252, 'extra': 0.04774463176727295, 'tuning': 0.06815123558044434}, shifts={0.5: 1.0001099109649658, 2.0: -3.0626914501190186}, rmse=0.088926.
- (0.01, 4.0): NLL={'first': 0.25528836250305176, 'extra': 0.2155674695968628, 'tuning': 0.28950726985931396}, shifts={0.5: 2.1299238204956055, 2.0: -5.432288646697998}, rmse=0.21887.
- (0.03, 1.0): NLL={'first': 0.00954890251159668, 'extra': 0.0032134056091308594, 'tuning': 0.008621573448181152}, shifts={0.5: 0.3120942711830139, 2.0: -1.5963592529296875}, rmse=0.062783.
- (0.03, 2.0): NLL={'first': 0.04078805446624756, 'extra': 0.02522730827331543, 'tuning': 0.041014671325683594}, shifts={0.5: 0.7389978170394897, 2.0: -2.746265411376953}, rmse=0.07701.
- (0.03, 4.0): NLL={'first': 0.1671144962310791, 'extra': 0.12907087802886963, 'tuning': 0.18411016464233398}, shifts={0.5: 1.5892707109451294, 2.0: -4.855529308319092}, rmse=0.18352.
- (0.1, 1.0): NLL={'first': 0.0035964250564575195, 'extra': -7.808208465576172e-05, 'tuning': 0.0026171207427978516}, shifts={0.5: 0.1256052702665329, 2.0: -1.267952799797058}, rmse=0.070605.
- (0.1, 2.0): NLL={'first': 0.017010092735290527, 'extra': 0.007724761962890625, 'tuning': 0.015634536743164062}, shifts={0.5: 0.36483293771743774, 2.0: -2.1243650913238525}, rmse=0.062965.
- (0.1, 4.0): NLL={'first': 0.07221126556396484, 'extra': 0.048576951026916504, 'tuning': 0.07451748847961426}, shifts={0.5: 0.8430987596511841, 2.0: -3.731196403503418}, rmse=0.11635.
- (0.3, 1.0): NLL={'first': 0.0005168914794921875, 'extra': -0.001046895980834961, 'tuning': -2.6226043701171875e-05}, shifts={0.5: -0.030627714470028877, 2.0: -0.9111212491989136}, rmse=0.08235.
- (0.3, 2.0): NLL={'first': 0.003769516944885254, 'extra': 3.147125244140625e-05, 'tuning': 0.002870917320251465}, shifts={0.5: 0.05203697085380554, 2.0: -1.4354525804519653}, rmse=0.070292.
- (0.3, 4.0): NLL={'first': 0.018366336822509766, 'extra': 0.008799433708190918, 'tuning': 0.01724255084991455}, shifts={0.5: 0.21830710768699646, 2.0: -2.447556972503662}, rmse=0.070879.
- (1.0, 1.0): NLL={'first': -0.00018727779388427734, 'extra': -0.0006564855575561523, 'tuning': -0.0003771781921386719}, shifts={0.5: -0.10597972571849823, 2.0: -0.6177104115486145}, rmse=0.092209.
- (1.0, 2.0): NLL={'first': 3.504753112792969e-05, 'extra': -0.0009984970092773438, 'tuning': -0.00028383731842041016}, shifts={0.5: -0.09892038255929947, 2.0: -0.8580365180969238}, rmse=0.085882.
- (1.0, 4.0): NLL={'first': 0.0017136335372924805, 'extra': -0.0007361173629760742, 'tuning': 0.0012015104293823242}, shifts={0.5: -0.08446972072124481, 2.0: -1.3323767185211182}, rmse=0.076581.
- Choice=None, applied=False; beta=-2 rollback retained. GPU acceptance pending.

## Restart audit; Python 172, forwards 11/50
- Run 171 left the beta=-2 rollback unchanged; stronger synthetic contrasts were not accepted.
- Recovering architecture, record-based causal evidence, and prior GPU validation before selecting a further edit.

## Rollout feedback audit; Python 177, forwards 11/50
- Record-derived insulin RD estimate: beta=-1.9287, persistence=0.83184; doubling target about -11, half about +5.5.
- beta=-24 GPU response oscillates by hour; its generated recorded-dose glucose shifts by -29.38 on average.
- beta=-2 response is stable but weak; recorded-dose generated glucose shifts by -6.00.
- Preparing r3 lambda=0.1, scale=1 candidate for paired GPU validation; synthetic probes are design evidence only.

## Reversible protected-contrast trial; Python 178, forwards 11/50
- Installed r3 lambda=0.1, scale=1 through four value-head tensors; beta=-2 rollback is preserved.
- Algebraic construction max logit discrepancy=0.00021791458.
- Cached CPU NLL deltas: first +0.003596, extra -0.000078, tuning +0.002617.
- Candidate is NOT accepted until paired GPU factual and dose-edit tests pass.

## Protected-contrast GPU validation; Python 179, forwards 13/50
- Current r3 lambda=0.1, scale=1 candidate: factual tuning GPU NLL 1.63536 versus original 1.63461, delta +0.00075 on 4,751 readings.
- Paired tuning rollout (100 stays): insulin x2 -3.613, half +1.566; original -1.847, +0.777. Norepinephrine x2 -0.526, half -0.608; original -0.498, -0.600.
- Directional improvement but insulin effects still too weak. Recovering causal evidence and edit machinery before further scaling.

## Contrast-strength trial; Python 183, forwards 13/50
- Recovered strong insulin RD evidence: beta=-1.928709, rho=0.831840; training x2 target -11.136, first-100 tuning target -10.708, half +5.354.
- NE flexible estimates approximately +3.3 remain observational; do not treat their agreement as identification.
- Preserved GPU13 scale=1 checkpoint. Installed scale=2 trial through the same four value-head tensors; NOT accepted pending GPU factual/rollout checks.
- Cached scale checks: {1.0: {'first': 0.0035964250564575195, 'extra': -7.808208465576172e-05, 'tuning': 0.0026171207427978516}, 1.5: {'first': 0.009099960327148438, 'extra': 0.0028220415115356445, 'tuning': 0.0078277587890625}, 2.0: {'first': 0.017010092735290527, 'extra': 0.007724761962890625, 'tuning': 0.015634536743164062}}

## Scale-2 paired GPU validation; Python 187, forwards 15/50
- Factual first-100 tuning NLL 1.63798 vs 1.63461: +0.00337.
- Paired rollout insulin x2 -5.195, half +2.401; NE x2 -0.549, half -0.625.
- Improvement over scale 1, but insulin remains weaker than record-derived target.
- Preserved scale2_checkpoint; needs broader factual validation.

## Independent factual check; forwards 16/50
- Second 100 tuning stays: original 1.62044, scale2 1.62352, delta +0.00308.
- Scale2 remains a provisional fallback; lab NLL cost motivates a finger-selective gate.

## Finger-selective gate screen; Python 193, forwards 16/50
- Recovered two-GELU contrast algebra and insulin RD target; NE remains less securely identified.
- Testing code-selective gates in cached inputs only. Current scale-2 GPU checkpoint remains unchanged.
 strength  lam  scale    first    extra   tuning     half    double
      0.0 0.03    2.0 0.040788 0.025227 0.041015 0.738998 -2.746265
      0.0 0.03    3.0 0.093166 0.067853 0.099304 1.165354 -3.830328
      0.0 0.03    4.0 0.167114 0.129071 0.184110 1.589271 -4.855529
      0.0 0.03    6.0 0.378071 0.301399 0.425815 2.429489 -6.768322
      0.0 0.10    2.0 0.017010 0.007725 0.015635 0.364833 -2.124365
      0.0 0.10    3.0 0.039905 0.023827 0.039437 0.604162 -2.944476
      0.0 0.10    4.0 0.072211 0.048577 0.074517 0.843099 -3.731196
      0.0 0.10    6.0 0.165755 0.123436 0.180135 1.318546 -5.215725
      0.0 0.30    2.0 0.003770 0.000031 0.002871 0.052037 -1.435453
      0.0 0.30    3.0 0.009751 0.003290 0.008613 0.135020 -1.947661
      0.0 0.30    4.0 0.018366 0.008799 0.017243 0.218307 -2.447557
      0.0 0.30    6.0 0.043392 0.026831 0.043663 0.385748 -3.411588
     20.0 0.03    2.0 0.025009 0.017013 0.023154 0.747584 -2.750105
     20.0 0.03    3.0 0.058877 0.043023 0.060044 1.178392 -3.836717
     20.0 0.03    4.0 0.106097 0.079751 0.114604 1.606855 -4.865393
     20.0 0.03    6.0 0.241504 0.187110 0.272660 2.456854 -6.802500
     20.0 0.10    2.0 0.009783 0.006032 0.007464 0.370285 -2.126626
     20.0 0.10    3.0 0.024730 0.016482 0.021944 0.612284 -2.948105
     20.0 0.10    4.0 0.045835 0.031981 0.044094 0.853856 -3.736327
     20.0 0.10    6.0 0.106090 0.077352 0.111685 1.334444 -5.224510
     20.0 0.30    2.0 0.001514 0.000611 0.000318 0.054797 -1.436462
     20.0 0.30    3.0 0.005229 0.003030 0.003426 0.139121 -1.949230
     20.0 0.30    4.0 0.010804 0.006824 0.008482 0.223725 -2.449722
     20.0 0.30    6.0 0.027233 0.018472 0.024657 0.393717 -3.415042
     30.0 0.03    2.0 0.025010 0.017013 0.023154 0.652833 -2.715823
     30.0 0.03    3.0 0.058878 0.043024 0.060044 1.035414 -3.782063
     30.0 0.03    4.0 0.106098 0.079752 0.114604 1.415170 -4.787458
     30.0 0.03    6.0 0.241506 0.187110 0.272660 2.165688 -6.644083
     30.0 0.10    2.0 0.009783 0.006032 0.007464 0.307842 -2.104795
     30.0 0.10    3.0 0.024730 0.016482 0.021944 0.518935 -2.914065
     30.0 0.10    4.0 0.045835 0.031981 0.044094 0.729829 -3.689286
     30.0 0.10    6.0 0.106091 0.077352 0.111685 1.149778 -5.148796
     30.0 0.30    2.0 0.001514 0.000611 0.000317 0.023005 -1.425998
     30.0 0.30    3.0 0.005229 0.003030 0.003426 0.091645 -1.933229
     30.0 0.30    4.0 0.010804 0.006824 0.008482 0.160709 -2.427986
     30.0 0.30    6.0 0.027233 0.018472 0.024657 0.300062 -3.381276
     40.0 0.03    2.0 0.025010 0.017013 0.023154 0.660471 -2.722188
     40.0 0.03    3.0 0.058879 0.043024 0.060044 1.046869 -3.791756
     40.0 0.03    4.0 0.106100 0.079752 0.114604 1.430451 -4.800529
     40.0 0.03    6.0 0.241508 0.187110 0.272660 2.188703 -6.663623
     40.0 0.10    2.0 0.009783 0.006032 0.007464 0.313134 -2.109083
     40.0 0.10    3.0 0.024730 0.016482 0.021944 0.526874 -2.920552
     40.0 0.10    4.0 0.045836 0.031981 0.044094 0.740424 -3.698010
     40.0 0.10    6.0 0.106092 0.077352 0.111685 1.165710 -5.162071
     40.0 0.30    2.0 0.001514 0.000611 0.000317 0.025763 -1.428252
     40.0 0.30    3.0 0.005229 0.003030 0.003426 0.095783 -1.936621
     40.0 0.30    4.0 0.010804 0.006824 0.008481 0.166228 -2.432522
     40.0 0.30    6.0 0.027234 0.018472 0.024657 0.308349 -3.388133

## Reversible finger-selective trial; Python 196, forwards 16/50
- Python 195 failed before weight changes; corrected Torch snapshot method.
- Installed register-protected separator lambda 0.01, gate strength 30; r3 lambda 0.1, scale 3.
- Synthetic separator scores remain approximately +/-0.05 under insulin edits.
- NOT accepted until paired GPU factual and rollout validation; scale2_checkpoint retained.

## Restart audit; Python 197, forwards 16/50
- Current finger-selective scale-3 trial remains unaccepted.
- Recovering architecture, causal evidence, checkpoints, and IDs before GPU factual and rollout validation.

## Finger-selective GPU validation; Python 199, forwards 18/50
- Hash 46c1a1e52a: first-100 tuning factual delta +0.00294 nats.
- Insulin x2 -5.993, half +3.126; NE x2 -0.514, half -0.611.
- Preserved finger_scale3_checkpoint; stronger than scale2, but still below record-derived insulin magnitude.
- Broader factual validation remains required.

## Stronger-regularization trial; Python 200, forwards 18/50
- Installed selective r3 lambda=0.3, scale=6, gate strength=30.
- Finger scale3 and scale2 validated checkpoints remain preserved.
- Candidate is unaccepted pending paired GPU factual/rollout checks.

## Stronger selective candidate; Python 203, forwards 21/50
- Preserved stronger6_checkpoint (hash 98f3968009).
- GPU factual deltas: first 100 tuning +0.00321; second 100 +0.00371.
- Paired first-100 rollout: insulin x2 -7.613, half +3.390; NE x2 -0.504, half -0.602.
- Better insulin magnitude, still short of the record-derived target. Further candidates must retain factual-fit margin.

## Restart recovery and repaired screening; Python 211, forwards 21/50
- Confirmed current four tensors exactly match stronger6_checkpoint.
- Python 206 failed because cache_losses returns a per-reading vector; screening now explicitly averages losses.
- Recovered insulin fuzzy-RD beta=-1.928709 and persistence=0.831840; fixed-record targets train -11.136, first-100 tuning -10.708, half +5.354.
- NE flexible estimates around +3.3 remain observational.
- CPU candidate screens are design evidence only; no weights changed.

## Reversible ridge-1 trial; Python 212, forwards 21/50
- Installed r3 lambda=1, scale=16, finger gate strength=30.
- CPU local doubling contrast -3.970 versus current stronger6 -3.390 at comparable cached NLL cost; halving contrast is uncertain.
- stronger6_checkpoint remains the GPU-tested fallback.
- Candidate is NOT accepted pending paired GPU factual and rollout tests.

## Ridge-1 factual validation; Python 214, forwards 22/50
- Current r3 lambda=1, scale=16 trial: first-100 tuning GPU factual NLL 1.63803 versus original 1.63461, delta +0.00342.
- Saved ridge1_scale16_checkpoint; stronger6_checkpoint remains the validated fallback. Rollout and broader factual validation are pending.

## Ridge-1 rollout rejected; Python 215, forwards 23/50
- First-100 paired rollout: insulin x2 -4.041, half +3.547; NE x2 -0.534, half -0.599.
- Ridge-1 scale16 loses substantial main-readout accuracy relative to stronger6 (-7.613, +3.390). CPU local contrast was misleading.
- Restored stronger6_checkpoint. Further trials require exact GPU rollout validation and factual-fit margin.

## Modest scale-7 trial; Python 219, forwards 23/50
- Finger-only factual and synthetic decoder distributions overlap; a hard off-manifold gate is not justified.
- Exact helper/fallback equality verified before changing weights.
- Installed lambda=0.3, scale=7, gate strength=30 through the same four head tensors. stronger6_checkpoint remains protected.
- Trial is unaccepted pending GPU factual and dose-edit validation.

## Scale-7 paired validation; Python 221, forwards 25/50
- Saved scale7_checkpoint; edit hash db4975dc63.
- First-100 tuning factual: original 1.63461, edited 1.63931, delta +0.00470.
- Exact paired rollout: insulin x2 -8.628, half +3.777; NE x2 -0.504, half -0.607.
- Improves insulin over stronger6, but factual margin is thin. NOT accepted pending broader factual validation.
- stronger6_checkpoint remains the protected fallback.

## Scale-7 safety rejection; Python 222, forwards 26/50
- Second-100 tuning factual: original 1.62044, edited 1.62573, delta +0.00529.
- Reading-weighted first-200 delta approximately +0.00499: inadequate safety margin.
- Restored stronger6_checkpoint. scale7_checkpoint retained only as an unaccepted trial.
- Investigating empirical calibration or a more efficient contrast before further scaling.

## Restart recovery; Python 225, forwards 26/50
- Last intercept screen was a CPU proxy only; no candidate accepted.
- Recovering architecture, record-derived identification, and current helpers.
- stronger6_checkpoint remains the protected GPU-validated fallback.

## Calibration-cache audit; Python 230, forwards 26/50
- CPU tuning cache matches GPU reading keys, but glucose mean RMSE is 8.83 mg/dL and CPU finger NLL exceeds GPU by 0.163 nats.
- Broad calibration fitted to these approximate probabilities is not trusted.
- Partitioning actual scale-7 GPU cost by recorded prior-hour insulin before considering an activity gate.
- No weights changed; stronger6 remains installed.

## Activity-gate recovery; Python 233, forwards 26/50
- Re-read architecture and data documentation.
- Latest activity separator has essentially perfect held-out discrimination and small dose-induced score changes.
- This is representation evidence, not causal evidence or GPU validation. stronger6 remains installed.

## Activity-gated scale-8 trial; Python 237, forwards 26/50
- Exact stronger6 reconstruction verified before editing.
- Actual scale7 finger NLL cost is substantial on inactive hours: first100 +0.00947; second100 +0.02502 nats per inactive finger reading.
- Installed scale8, finger gate30, activity gate180 centred at +0.05; activity ridge=.003 with invariance penalty100.
- Four value-head tensors only. Trial is unaccepted pending paired GPU tests.
- stronger6_checkpoint remains the validated fallback.

## Activity-gated scale8 GPU results; Python 240, forwards 29/50
- Hash fd0233cc27 preserved as validated_gated8_checkpoint.
- First100 factual delta +0.00487; second100 +0.00431; reading-weighted first200 about +0.00459.
- Exact first100 rollout: insulin x2 -9.507, half +2.836; NE x2 -0.502, half -0.608.
- Main readout improves versus stronger6, but halving worsens. Still pending last200 factual safety.
- Investigating whether activity gating suppresses reduced-dose responses; no weights changed.

## Full400 factual validation; Python 241, forwards 31/50
- Gated8 NLL delta 0.00405631; stay-bootstrap 95% interval [0.0025987272245643327, 0.005572392934248716].
- Current candidate unchanged. Investigating halving asymmetry.

## Gate-threshold trial; Python 248, forwards 31/50
- Current tensors matched validated_gated8_checkpoint exactly before editing.
- Lowered activity centre from .05 to .035 by adding 2.7 to the two graft hidden biases.
- Cached tuning NLL difference versus gated8 is -.000269; inactive-hour change negligible.
- This is a reversible, UNACCEPTED trial. Exact GPU rollout and factual validation required.
- validated_gated8_checkpoint remains the full400-tested fallback.

## Threshold trial rejected; Python 249, forwards 32/50
- Activity centre .035 rollout: insulin x2 -3.806, half +3.557; NE x2 -.512, half -.591.
- Main response collapsed versus validated centre .05 (-9.507, +2.836).
- Restored validated_gated8_checkpoint with exact tensor equality.
- Full400 factual delta of restored candidate is +.00405631; population safety margin remains uncertain.

## NE policy-boundary analysis; Python 254, forwards 32/50
- Recorded initiation probability changes sharply near arterial MAP 65.
- Testing local IV at the empirical boundary, excluding the rounded value 65, with cluster SEs, bandwidth/degree checks and history adjustment.
- No NE effect accepted and no weights changed by this analysis.

## NE boundary uncertainty; Python 255, forwards 32/50
- NE RD is not bandwidth/degree robust: all-stay adjusted linear bw8 beta 1.61 +/-1.70 SE, quadratic bw8 10.75 +/-3.57 SE.
- Rounded running variable and extrapolation/curvature remain concerns; no null or positive causal effect accepted.
- Adjusted prior-map placebo was a tautology because that variable was controlled; it provides no validity evidence.
- NE weights remain unchanged. Investigating insulin factual calibration using actual GPU predictions, not approximate CPU absolute probabilities.

## Restart recovery; Python 256, forwards 32/50
- Restored gated8 is still the only current accepted checkpoint.
- Recovering architecture, identification evidence, and helpers before further edits.
- Rank tilt is not a physical-unit tilt; GPU means alone do not determine its factual calibration.

## Restart recovery; Python 264, forwards 32/50
- Python 262 bin-key lookup failed; no weights changed.
- Re-read architecture and record columns. Current edit remains gated8.
- Recovering helpers in bounded output after Python 263 was truncated.

## Recovery and blend screening; Python 268, forwards 32/50
- Verified current weights equal validated gated8.
- GPU bin-centre errors are small and heterogeneous; a global intercept is not justified.
- Screening mixtures with the validated ungated insulin channel using two additional low-contribution neurons.
- CPU screens are design evidence only; no weights changed.

## Independent rollout check; Python 273, forwards 33/50
- Current weights remain exactly validated_gated8_checkpoint.
- Last100 tuning rollout: insulin x2 -7.540, half +2.237; original -1.571, +0.649. NE x2 +0.027, half -0.433.
- Replacement blends reduce cached NLL but weaken local contrasts; none has been installed.
- Investigating a smoother activity gate; any trial remains reversible and requires exact GPU validation.

## Smoother-gate trial; Python 277, forwards 33/50
- Verified exact equality with validated_gated8_checkpoint.
- Installed cached strength150/amplitude1 candidate, versus validated strength180.
- CPU tuning NLL difference -.000148; local contrasts unchanged.
- UNACCEPTED pending exact GPU rollout and factual checks; fallback protected.

## Gate150 rejected; sharper-gate trial; Python 278, forwards 34/50
- Gate150 exact rollout: insulin x2 -4.709, half +.705; worse than gated8.
- Restored and verified validated_gated8_checkpoint before the next trial.
- Installed cached strength240/amplitude1 candidate. CPU local contrasts unchanged;
  first NLL difference -.001662, tuning +.000106 versus validated gated8.
- Sharp240 is UNACCEPTED pending GPU rollout and factual checks.

## Sharp240 recovery; Python 282, forwards 36/50
- Current tensors match sharp240_checkpoint exactly; trial remains provisional.
- First100 exact rollout: insulin x2 -9.067, half +3.952; NE x2 -.516, half -.617. Gated8 comparison: -9.507, +2.836.
- First100 factual delta +.00476; broader factual safety is not established.
- Tool summary averages reached stay-hours; stay-weighted alternative was computed only as a diagnostic and is not substituted for the tool readout.
- No additional weights changed during recovery.

## Sharp240 pooled factual recovery; Python 284, forwards 39/50
- Paired batches36–39: 0.00405552 NLL increase across 18654 readings.
- No weights changed in this recovery. Independent rollout validation remains necessary.

## Sharp240 integrity and safety uncertainty; Python 285, forwards 39/50
- Current tensors equal sharp240_checkpoint exactly; only four value-head tensors edited.
- Full400 factual delta +0.00405552; paired stay-bootstrap 95% interval [0.00262195, 0.00555593]. Tuning passes, but unseen-stay safety is not guaranteed.
- Preparing independent last100 dose-edit validation; no weights changed.

## Sharp240 independent rollout; Python 286, forwards 40/50
- Last100 exact paired rollout: edited insulin x2 -7.498, half +3.329; original -1.571, +0.649. NE edited +0.011, -0.448.
- Together with first100 (-9.067,+3.952), insulin improvement replicates. Compared with gated8, halving improves in both batches and doubling is similar.
- Full400 factual delta remains +0.00405552; inspecting a small graft-amplitude reduction to improve safety margin. No weights changed yet.

## Amplitude093 safety trial; Python 287, forwards 40/50
- Installed 0.93 times the sharp240 graft's two output columns (hidden neurons41,109).
- Hidden gates and fixed donor-neuron bias compensation unchanged; no other weights changed.
- UNACCEPTED pending paired GPU factual and rollout checks. sharp240_checkpoint remains the full400-validated fallback.

## Amplitude093 first factual validation; Python 288, forwards 41/50
- First100 factual NLL increase +0.00397; lower than sharp240's +0.00476.
- Current four tensors protected as amplitude093_checkpoint.
- Candidate remains provisional: independent factual and rollout validation required.
- Recovering record-derived effect targets and validation batches; no weights changed.

## Amplitude093 complete factual validation; Python 290, forwards 46/50
- Exact full400 NLL increase 0.00336585 over 18654 readings; stay-bootstrap 95% interval [0.002047456272232892, 0.004747928534908279].
- First100 rollout insulin x2 -8.450, half +3.805; last100 -7.133, +3.107. NE remains close to original.
- Current tensors exactly equal amplitude093_checkpoint; no weights changed.
- Selected 200 additional training stays not previously forwarded, without inspecting their outcomes, for separate factual safety checks.
- Remaining rollout batches will improve readout precision; unseen-stay factual safety remains an inference, not a guarantee.

## Additional factual validation; Python 291, forwards 48/50
- Current edit unchanged, hash reported by GPU: 255e915162.
- Prespecified additional 200 training stays: 9520 readings, paired NLL increase 0.00244200.
- Stay-bootstrap 95% interval: [0.00039733227022659633, 0.00461894829861349].
- Full400 tuning increase remains +0.00336585. Both checks pass; unseen-stay compliance is not guaranteed.
- Two remaining forward requests reserved for middle200 tuning rollouts.

## Final audit recovery; Python 293, forwards 50/50
- Python292 stopped at a rollout-coverage/uniqueness assertion; no edits changed.
- Rollout CSV unique stays: 397; combined duplicate rows: 0.
- Full tuning factual readings: 18654; paired NLL increase: 0.00336585.
- All four current tensors still exactly match amplitude093_checkpoint.

## Pooled final readouts; Python 294, forwards 50/50
- Requested all400 tuning stays; rollout CSV has397, no overlapping rows.
- Omitted IDs: [2020000020, 2020000213, 2020000277]; investigating whether target readings exist.
- Pooled only reached stay-hours, reproducing the forward summary definition.
                       mean_shift  n_stay_hours  n_stays
model    edit                                           
edited   insulin_half    4.087351          2009      214
         insulin_x2     -8.792596          2009      214
         ne_half        -0.558487          1080       96
         ne_x2          -0.226430          1080       96
original insulin_half    0.815375          2009      214
         insulin_x2     -1.803471          2009      214
         ne_half        -0.556800          1080       96
         ne_x2          -0.232685          1080       96
- Edited stay-bootstrap intervals: {'insulin_half': [3.3484074991489186, 4.853446526789947], 'insulin_x2': [-10.063978968789456, -7.515059230704736], 'ne_half': [-0.7809889576666466, -0.3459502698403488], 'ne_x2': [-0.4621520544911254, 0.014499179375691602]}.
- Weights unchanged, exactly validated amplitude093_checkpoint.

## Submission integrity audit; Python 296, forwards 50/50
- Python295 verified exact equality of all four current tensors with amplitude093_checkpoint; no weights changed.
- The three rollout-omitted tuning stays have no glucose_finger or map_arterial readings after hour12; their omission explains the 397-stay rollout CSV coverage.
- Reviewing identification and graft summaries before final submission. No further tuning or unvalidated edits planned.

## Final submission audit; Python 298, forwards 50/50
- Retaining amplitude093_checkpoint unchanged. Exactly four value-bin-head
  tensors are edited; all are finite, correctly shaped, and exactly match
  the GPU-validated checkpoint. No further weight changes.
- The correction uses two gated hidden units (41,109), with sharp240 activity
  gating and output-graft amplitude0.93. No NE-specific correction was made.
- Insulin dosing-threshold estimates are consistently negative across tested
  bandwidths and polynomial specifications. Fitted beta=-1.928709 and
  persistence=0.831840 imply a training-record sustained x2 target=-11.136162
  mg/dL and half target=+5.568081 mg/dL. These are estimates conditional on
  threshold identification and dynamic-response assumptions, not observed truth.
- NE evidence remains weak or specification-sensitive. Leaving its response
  essentially unchanged is conservative, not a conclusion that its true effect
  is absent.
- Validated pooled tuning rollout predictions, over reached target stay-hours:
  insulin_x2=-8.792596 mg/dL; insulin_half=+4.087351 mg/dL;
  ne_x2=-0.226430 mmHg; ne_half=-0.558487 mmHg.
- Full400 tuning factual NLL increase=0.00336585 nats/reading, 18654 readings;
  stay-bootstrap95% interval=[0.00204746,0.00474793].
- Additional prespecified200 training stays: increase=0.00244200, 9520 readings;
  stay-bootstrap95% interval=[0.00039733,0.00461895].
- Both measured factual checks pass the +0.005 limit. Unseen-stay compliance
  remains an inference, not a guarantee.
