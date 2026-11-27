# Notebook
## Facts (runs 1-5)
- Stays: hosp adm -2h, ICU adm 0h, end 25.1667h. Readings at integer hours; insulin/NE records at h+0.0833, run 1h.
- Insulin rate grid 0.5 u/h (0.5..14.5), dose=rate. NE rate 0.01..1.00 (cap 1.00, 33k recs), dose=4.8*rate.
- 18335/30000 stays have insulin, 13174 have NE. NE x2 pushes rates >1 (never seen in training, bin 16).
- Grid saved /work/grid.pkl: (stays_tr,Gtr,stays_tu,Gtu), Gtr[key] DataFrame stay x hour(-1..25), keys gf gl ma mc ins ne

## Findings (runs 6-25, fwd 1)
- Insulin protocol: sliding scale on the finger reading that hour: <150 none; 150-180 base 1; 180-220 base 2; 220-300 base 3; 300+ ~6 u/h.
  rate = base * random multiplier (log sd ~0.33), rounded to 0.5. Multiplier: no autocorr (0.008), no stay effect, unrelated to covariates
  -> within-tier dose variation is as good as random given gf_now. Finger readings hourly if last >=150, else ~1/3 per hour; lab random ~20%/h.
  finger and lab measure same glucose (mean diff 0, noise sd grows with level).
- NE protocol: titration. Start when MAP<65; continue 97.5%; adjust from previous rate: up if MAP<65, hold 65-75, down >75; big noise in steps.
- Insulin effect (local projections, FE on gf_t, within tier, deconvolved for protocol feedback): direct response of gf to 1 u/h at lag 1..10 h:
  finger-based [-1.99 -2.37 -2.01 -1.65 -1.12 -0.89 -0.61 -0.47 -0.26 -0.11] (sum -11.5); lab-based similar (sum -12.6). Same across tiers (linear, level-independent).
- dose_edit tool: rows = hours 13-24 with a recorded outcome reading; reached = hour > first drug hour >=12 (verified exactly).
- fwd 1 (original, tuning sel1 = /work/sel1.json, 100 stays): ins x2 -1.60, ins x1/2 +0.70, NE x2 -0.15, NE x1/2 -0.40.
  My truth estimate on same rows: ins x2 ~ -13.3..-14.3, x1/2 ~ +6.7..+7.1. Model underestimates insulin ~8x.
- Plan: re-implement model locally in torch, validate against 'states'/'factual' outputs, then design edit.

- NE effect on map_arterial (LP, FE band x prev rate, deconvolved): direct per 1 mcg/kg/min, lags1-6: [3.36 2.55 1.84 0.94 0.50 0.20], ~0 after; sum ~9.4. lag-1 effect same at low/high rates (linear).
  Titration uses arterial MAP band if present (else cuff): <65 up ~+0.12 log, 65-75 hold, >75 down ~-0.2; noise sd ~0.3 log. Arterial line in ~48% stays (95% hours, random).
- Truth est on sel1 rows: ins x2 -13.3..-14.3, half +6.7..+7.1; NE x2 +4.53, half -2.27. Model: -1.60, +0.70, -0.15, -0.40.
- Model's insulin shift ramps slowly (-0.4 at 1h ... -3 at 12h after first dose) vs truth (-3, -6.8, -9.7, -11.9 ... -20 at 12h). NE model ~0 flat.
- fwd 2: states of original on sel1 in /work/fwd_2 (5116 bundles). Bundles = distinct times; END atoms at record end (s+1.0833); final bundle 25.1667 (admission ENDs). has_insulin = new insulin record.

## Runs 39-43 (after restart 1)
- Old local-reimplementation function sources are lost (no getsource). VOCAB has no categorical vocabularies -> exact re-implementation is guesswork.
- States (fwd_2, sel1): x = [out 0:192 | h 4x192 192:960 | zpost 960:992]. Within-tier insulin dose is decodable from state: CV R2 out .54, h .67, all .68.
- NEW PLAN: head-level edit (value head + transition table), driven by a dose readout from the state. A head-only edit leaves factual states unchanged
  -> factual NLL can be computed offline once the local head matches the factual tool output. Correction must use dose minus protocol-expected dose
  (otherwise it double-counts the average effect already absorbed by the model).

- Run 46-47: TRUE one-step coefs (gf_{t+1} on gf_t tier-FE quad, gf_{t-1}, lab_t; train). Raw dose lags1-6: [-2.01 -.74 1.40 1.28 1.22 1.15] (lags>=3 confounded).
  On dose DEVIATION (dose - E[dose|tier], tab E[dose|tier0..4,miss]=[0,1.03,2.07,3.24,6.24,.03]): [-2.01 -0.63 0.07 0.48 0.57 0.37]. Resid sd 12.8.
- fwd 3: factual original on tuning 2020000000..99: value NLL 1.63461 (n 4751); gf 1.4639, gl 1.2941, ma 1.4712, mc 2.012. File /work/fwd_3/factual_original.csv

## Runs 63-73 (after restart 2): value head reconstructed exactly (code in /work/lib.py)
- Codes: mc 5, gf 6, ma 7, ins 8, ne 9, gl 10. Value head = MLP 192-192(GELU)-17 shared across codes + table embed_transition.value_bin[code*17+last bin].
- with_code = query_norm(view + type_emb[1]) + embed.code[c] + embed.bins.value[last bin] + last_numeric(f) ; f: col0 has_last, col3 log1p(age_s)/10, col6 z_last (clip +-10).
- view = out + out_norm(mean_slots(h*exp(-dt*exp(flow.a)))) + gap(log(dt_s)/10) + z_head(z), z ~ prior (mu=first 32, logsd=last 32), 4 draws averaged.
- Local predicted means match tool within noise floor (MAE 1.32 vs floor 1.28) on 38 stays overlapping sel1 (states fwd_2) and fwd_3 (factual).
- Glucose centres 73..272, only 1.9% of h13-24 gf readings above top edge 260. NE rate z=(r-.46)/.59, train max r=1.0 (bin 12); NE x2 -> r up to 2 (bin 16, z 2.6: out of training range).
- Insulin dose/rate z=(d-1.5)/1.0; bins: d>5 -> bin 16.

## Runs 85-90 (after restart 3)
- Previous reply was cut by a safety filter; kept strictly to allowed methods (records/state regressions, closed-form edits, tool checks).
- Kernel still holds: X (sel1 states 5116x992), RR (10353 reading rows sel1 with local pm), WC/HID/LG (8 z-draws), Dtr2 (train one-step design),
  build_ins_design2(G, stays) -> design (s = position in stays list!), ins_regress2(D, ycol) -> (coef, se), layout 15 tier coefs + 10 extras + dev1..6.
- One-step dev coefs (same spec, extras): TRUE train [-2.04 -1.23 -0.25 0.38 0.80 0.69] (se .045); MODEL pm on 200 tuning stays (f3 + sel1)
  [-0.34 0.13 0.26 0.11 0.12 -0.10] (se .12). Delta = [-1.71 -1.36 -0.51 0.27 0.68 0.79]. Model tier slopes on g0 weaker (0.97/0.96/0.84 vs ~1.05).

- Runs 93-95: batch0 (fwd_5, 100 train stays both drugs): model dev coefs [-0.39 .24 .08 .11 .22 .34] (se .10) vs truth [-2.04 -1.23 -.25 .38 .80 .69].
  dev1 decodable from head q only CV R2 ~0.22-0.27 (out_prev .28, full state .45); dev2 ~0.05. Model's -0.39 ~= true*R2 -> representational bottleneck.
  Plan A: lag-1 unit-gain readout correction in value head (2 GELU units per drug, code-gated), coef ~ -2.25 (to match total sum of one-step coefs), cost ~0.003.
- fwd_6: by mistake states for 100 hand-typed train stays (not batch1); saved in /work/states_stays.json. Use batches from /work/sel_tr2.json for next requests.

## Runs 105-131 (after restart 4)
- fwd_4=trA_A (sel_train.json A), fwd_5=tr2_0 (=batch0), fwd_6=hand-typed (states_stays.json), fwd_7..9=tr2_1..3. All 'states' of original.
- Kernel: datasets[0..4]=(states, rows) for fwd_5..9; base_out[i]=(pm,nll,P,hidden16) of original head; head_run(Xs,R,Wd) = local value head (16 z-draws, matches tool).
- Tensor names: W0=attr_heads.value_bin.0.weight, b0=attr_heads.value_bin.0.bias, W2=attr_heads.value_bin.2.weight, TR=embed_transition.value_bin.weight. ORIG dict is float64.
- Old ungated make_edit (a_ins=-1.9,a_ne=3) costs 0.018 nats/reading: noise on glucose rows w/o insulin in prev bundle + cuff MAP. Rejected.
- Per-draw head inputs: head_run with W0=I, b0=100 -> hidden=wc+100. WCall(16,27963,192) for datasets 0-4; A_ has code, hi/hn (drug in prev bundle), dev1, nedev1, stay.
- Readout facts: dose info sits in low-variance directions of the head input; target-only ridge readouts explode off-target (+-130). Fix = generalized ridge with off-target penalty (lam 0.01) + code-embedding-invariant weights.
- NEW EDIT gated3(a_ins,a_ne): units [26,41,131,159]. Insulin pair u1-u2 = unit-gain readout of dev1 (ROi5, R2 .52) gated by glucose code gate (gvec,K=40) x insulin-in-prev-bundle gate (Gi3,A=30,M=-10).
  NE pair u3-u4 = readout of nedev1 (ROn4, R2 .19) gated to map_arterial only (Gn2, B=10, M=-1). Logit directions tg/tm. Pair outputs verified exact on all 16 draws; max pre-act 277.
- Offline cost a_ins=-1.9,a_ne=3: 0.0052/reading (ma on-rows .013 -> ~.0035 of total; gf .0029 on rows). Lag-1 slopes: gf -2.17/u/h dev, gl -1.44, ma +3.27 per unit NE-rate dev.
- Next: fwd dose_edit both on sel1 with gated3(-1.9,3) to calibrate a_ins, a_ne against truth est (ins x2 -13.3..-14.3, half +6.7..7.1; NE x2 +4.53, half -2.27).


## Runs 176-188 (after restart 5)
- BUGFIX: global `lin` had been overwritten by a 1-arg lambda (kept as lin_r); restored lin(x,name)=x@W.T+b. eval_edit/head_run work again.
- gated3(a_ins,a_ne) returns dict {'W0','b0','W2','TR'} (short keys; KMAP maps to tensor names). Saved edit == gated3(-1.9,3) (hash 262704e88f).
  It repurposes 4 head units; gated3(0,0) != original (costs ~0.0004-0.0013 by itself).
- fwd_10: dose_edit both, tuning f3 (2020000000..99), edit gated3(-1.9,3). orig: ins x2 -1.87, half +.76, NE x2 -.51, half -.70.
  edited: -9.85, +4.13, -0.80, -1.91. MY TRUTH on same rows (truth_ins/truth_ne): -13.38, +6.69, +4.89, -2.45.
  Profile by k=hours since first reached: truth -3.0,-6.8,-9.6,-11.9,-13.6,-14.9,-16.2,-16.4,-17.3,-17.5,-18.3,-20.3; edited -3.35,-4.6,-5.7,-7.5,-8.1,-9.0,-10.6,-11.2,-13.4,-14.5,-16.7,-18.9.
  -> edited ramps ~linearly, truth front-loaded. Only the mean is judged. Linear scaling suggests a_ins ~ -2.7 for x2.
  NE x2 edit fails (rates >1 are OOD bins 13-16; bins tables embed.bins.rate etc are SHARED across codes, insulin uses bins 13-16 -> no free fix).
- fwd_11: factual both on f3: orig 1.63461 -> edited 1.63798 (+0.0034). per-row: gf +.0062 (n1538), gl +.0013 (488), ma +.0053 (1109), mc 0.
- RR = 5214 sel1 rows (src '1', states X=fwd_2) + 5139 train-A rows (src 'A', states NOT in X!). Use RR[src=='1'] with X only.
- Offline head_run on sel1 rows validated vs tool on 38 overlapping stays (corr dpm .975-.986, dn similar).
- Offline OUT-OF-SAMPLE (sel1) cost of saved edit vs orig: +0.0080/reading (gf +.0056/row, ma +.0216/row, gl -.0008). sel1 is drug-heavy;
  random f3 gave +.0034 in tool. eval_edit (train datasets, in-sample readouts) is optimistic. Factual cost depends on drug-row share -> risky.
- NEXT: decompose cost (insulin vs NE part) per row type on sel1; reweight to random-stay composition; pick a_ins/a_ne; check with tool on all 400 tuning stays.

- Run 194: ins_only(a)=gated3(a,0) with NE units 131,159 RESTORED to original. sel1 offline cost: a=-2.2 .0025, -2.4 .0031, -2.6 .0037, -2.8 .0044 (6D rows +.010..+.017). Non-glucose rows ~0.
- NE part DROPPED (noisy readout R2 .19, gate leaks, cost ~.024 per NE-row, cannot fix x2 OOD). Saved ins_only(-2.6) (TR edit dropped). Next: dose_edit both on f3 to compare with fwd_10 (-1.9/3 -> -9.85).

- fwd_12: dose_edit both on f3 (2020000000..99) with ins_only(-2.6) [hash 924a84db1e]: orig x2 -1.85, half +.78; edited x2 -13.01 (truth -13.18 -> ratio .98), half +5.41 (truth 6.59, ratio .80); NE unchanged (-.51/-.62 vs truth 4.76/-2.38). (fwd_10 was on sel10, a different set!)
- Run 198: TRUE insulin effect linear: dev1 pos slope -2.08 (.07), neg -1.99 (.10), quad -.055 (.03). Truth x2/half ratio = -2; model asymmetry 2.4 is artifact.
- Next: factual both at -2.6 on all 400 tuning stays (sel_f3 + /work/sel_tu0..2.json).

- fwd_13: factual both tu0 (2020000100..199) with ins_only(-2.6): +0.00427/reading (gf +.0126/row, gl +.0061, ma/mc ~0). Too close to .005.
- Runs 200-206: persistence check on random stays: within-tier resid slope on prev gf only +2.2/50mg (se .78) -> dropped. Readout not overfit (in-sample corr .67 vs sel1 .72). Edited one-step slope ~-4 vs truth -2 (model lacks dose memory beyond lag1; lag-1 overshoot needed for sustained effect).
- Saved ins_only(-2.5). Plan: factual both on all 4 tuning batches at -2.5; dose_edit at -2.5.

## Runs 207-209 (current state, edit = ins_only(-2.5), hash 6ace05eef2)
- Tool factual at -2.5: f3 (2020000000..99) +0.00284 (gf +.0087/row, gl +.0005); tu1 (2020000200..299) +0.00369 (gf +.0108, gl +.0056).
  tu0 (2020000100..199) at -2.6 was +0.00427 (est ~.0039 at -2.5). Mean ~.0035, spread +-.0005. Budget .005.
- Cost decomposition on tool rows: cost from one-step overshoot (gain dpm/dev ~-2.9 vs factual optimum ~-1) + readout noise; no separable failure mode.
- Remaining plan: dose_edit at -2.5 (expect x2 ~-12.5 vs truth ~-13.2), factual tu0/tu2 at -2.5, then FINAL.
- FINAL-ANSWER facts: true insulin IRF per 1 u/h lags1-10 [-1.99 -2.37 -2.01 -1.65 -1.12 -0.89 -0.61 -0.47 -0.26 -0.11] (sum -11.5), linear;
  true NE IRF per 1 mcg/kg/min lags1-6 [3.36 2.55 1.84 0.94 .50 .20] (sum 9.4). Truth on tuning rows: ins x2 ~-13.2..-13.4, half ~+6.6..6.7, NE x2 ~+4.8..4.9, half ~-2.4..-2.45.
  Original model: ins x2 ~-1.85, half ~+.77, NE x2 ~-.5, half ~-.6..-.7. NE not edited (noisy readout, OOD rates>1, cost).

## Run 211 (after restart 6)
- fwd_16: dose_edit both on f3 (2020000000..99) with ins_only(-2.5) [hash 6ace05eef2]: edited ins x2 -12.57 (truth est ~-13.2), half +5.21 (truth est ~+6.6);
  NE unchanged -.52/-.62 (truth est +4.8/-2.4). orig -1.85/+.78/-.50/-.60.
  Errors now: ins x2 ~.6, ins half ~1.4, NE x2 ~5.3, NE half ~1.8 -> NE is the largest remaining error; factual budget mostly used (~.0035 of .005).
- Reply after fwd_16 was cut by safety filter; continuing with allowed methods only (record/state regressions, closed-form edits, tool checks).

## Runs 212-227 (restart 6)
- fwd_10 (gated3 NE a_ne=3) by NE rate: 62% of NE reached rows have rate in (.75,1.0] (train NE rate capped at 1.0, 12% of MA rows at exactly 1.0).
  NE readout response wrong-signed at low rates (x2 -.92, half +.80), right for half at high rates (-2.28), x2 at high rates ~0 (OOD 2.0).
- Raw NE-rate readout from head input (MA rows, ridge, CV R2 .51) SATURATES: pred .035 (off), .35 (r~.05), .43, .51, .59, .62, .63 (r=1.0), sd .14.
  -> head input encodes NE presence, hardly NE level/deviation. Any NE head readout = mostly noise. NE abandoned for good (needs core edits; no budget).
- Insulin asymmetry (fwd_16, -2.5): by rate (0,1]: x2 -2.1 half -.6; (1,2]: -12.9/+2.8; (2,3]: -24.7/+16.9. Gate (units 26/41 sum) always on for hi rows (121..194), off (-450..-260) else.
  Readout term vs dev1 slope pos .56, neg .44 (mild). Negative-side boost would trade x2 vs half error at ~same cost -> not done.
- Plan: compare a_ins -2.6 (x2 -13.01, half 5.41 on f3, fwd_12) vs -2.5. Measure factual at -2.6 on tu2, f3, tu1 (tu0 was +.00427).

- Run 228: saved ins_only(-2.6) [hash 924a84db1e] (W0,b0 same as -2.5; W2 differs). TR original.
- fwd_17 factual tu2 (2020000300..399) at -2.6: +0.00100 (gf +.0058/row, gl -.0043). Batch spread large: tu0 .00428 (-2.6), f3 .00283, tu1 .00369 (-2.5).
- Run 230 bootstrap over 400 tuning stays: -2.6 mean .00314, P(>.005) N=100 10%, N=200 4%, N=400 .65%; -2.5 mean .0028 (P 4.7%/1.25%/.08%).
- Run 231: cost NOT from outliers (51% from |dpm| 2-4 rows); residual slope on dpm .156 (factual optimum gain ~.16x current). No cheap cost cut.
- Run 232: /work/sel_cost.json = 4 batches x 100 unused train stays. fwd_18 (batch0) at -2.6: +0.00164 (gf +.0052/row).
- fwd_19/20/21 (train batches 1-3 at -2.6): +.00261, +.00368, +.00104. 8 batches (800 stays) at -2.6: .00428,.00317*,.00413*,.00100,.00164,.00261,.00368,.00104 (*scaled from -2.5 x1.12). Mean ~.0027.
- dose_edit at -2.6 on all 4 tuning batches (fwd_12 f3, fwd_22 tu1, fwd_23 tu0, fwd_24 tu2). Pooled (edited/orig/truth):
                 n  edited   orig   truth  ratio
edit                                            
insulin_half  2009   5.839  0.815   6.773  0.862
insulin_x2    2009 -12.356 -1.803 -13.545  0.912
ne_half       1080  -0.558 -0.557  -2.264  0.247
ne_x2         1080  -0.230 -0.233   4.528 -0.051
- Pooled tuning (400 stays) at -2.6: ins x2 -12.36 vs truth -13.55 (ratio .912), half 5.84 vs 6.77 (.862); NE -0.23/-0.56 vs 4.53/-2.26. f3 was a lucky batch.
- fwd_25: factual tu0 at -2.8 (hash c4297930e4): +0.00492 (vs .00428 at -2.6 -> factor 1.15 per 0.2). Scaled bootstrap: -2.7 mean ~.0029, P(>.005) N=400 ~.5%; -2.8 ~1.3%.
- Run 240: chose and saved ins_only(-2.7) (W2 only differs; W0,b0 as before; TR original). Expected pooled: x2 ratio ~.95, half ~.90.

## Runs 241-248 (restart 7)
- fwd_26 = factual tu0 (2020000100..199) at ins_only(-2.7): +0.00459 (gf +.0136/row, gl +.0065). Consistent with interpolation.
- gated3(a,a_ne) == ins_only(a) + units 131/159 only (TR original). head_run(X,R1,Wd) returns (pm,nll,P,H[16,N,192]); needs float64; honours 'b2'.
- Offline sel1 test-half: ins_only(0) cost 0 (repurposing free); ins_only(-2.7) .0045; gated3(-2.7,2) .0060 (NE extra .0015, ma +.0057/row). NE extra vs a_ne (all sel1): a1 .0006, a2 .0027, a3 .0063 (~quadratic).
- Compensation regression for lost units: useless (R2 .28), dropped.
- Est. random-stay NE cost: a2 ~.0007, a1.5 ~.0004. Testing gated3(-2.7,2) dose_edit on tu1 (compare fwd_22 = ins_only(-2.6) on tu1).
- fwd_27: dose_edit tu1 with gated3(-2.7,2) (reached rows): ins x2 -12.93 (truth -14.67, .88; ins_only(-2.6) -12.47), half 7.50 (truth 7.34, 1.02; -2.6: 7.19);
  NE half -1.10 (orig -.54, truth -2.55), NE x2 +.26 (orig -.11, truth 5.10). NE pair helps both, but...
- fwd_28: factual tu0 with gated3(-2.7,2): +.00827 vs ins_only(-2.7) +.00459 -> NE pair costs +.0037 (ma +.008/row, gf/gl +.0026/row leak). 5x offline estimate. NE DROPPED FOR GOOD.
- Restored ins_only(-2.7) (units 131/159 original). Next: factual tu1 at -2.7 (confirm hash 2bc751d207, cost; tu1 at -2.5 was .00369). If >.005 -> step back to -2.6.

## Runs 252-258 (restart 7, FINAL)
- Kernel restarted (OOM) at run 253; helper functions lost; saved edit on disk intact = ins_only(-2.7) (W0/b0 rows 26/41, W2 cols 26/41 = +-v, norm 3.83, scale with gain).
- Factual cost at -2.7 (direct or gain^2-scaled), 14 batches x100: tuning .0046 .0043 .0033 .0011; old train .0018 .0028 .0040 .0011; fresh train (fwd_30-35) .0027 .0032 .0010 .0002 .0026 .0028. Pooled mean ~.0025, tuning-only .0033, batch sd .0013.
- Bootstrap P(>.005) at -2.7: all-pool N=400 .2%; tuning-only N=400 1.45% (+mean uncertainty -> few %). Kept -2.7 (higher gain ~ same expected value, more risk).
- dose_edit at -2.7 on all 4 tuning batches (fwd_36-39) pooled:
model         edited  original   truth  ratio_edited
edit                                                
insulin_half   6.053     0.815   6.773         0.894
insulin_x2   -12.758    -1.803 -13.545         0.942
ne_half       -0.559    -0.557  -2.264         0.247
ne_x2         -0.227    -0.233   4.528        -0.050
- FINAL: ins_only(-2.7), hash 2bc751d207. NE not edited.
