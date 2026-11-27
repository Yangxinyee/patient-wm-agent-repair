# Notebook (ICU world-model dose repair)

## Data facts (runs 1-15)
- All stays 25.17 h; readings at integer hours (-1..24), drug records at h+0.0833 lasting 1 h.
- Insulin policy (deterministic given reading): uses finger glucose at hour h (else lab if no finger). No insulin if <150;
  tiers base rate 1/2/3/4(~4.2) U/h at thresholds 150/180/220/260 (50/50 at exact threshold values).
  Rate = base * lognormal noise (log-sd ~0.3) rounded to 0.5. Noise is iid (lag1 corr 0.004, no stay effect,
  no corr with pre-treatment vars). ALL 'skips' are at hour -1 (pre-ICU) -> skip variation is confounded, don't use.
- NE: capped at 1.0 mcg/kg/min (13.6% at cap); given ~always when MAP<65; analysis pending.
## Insulin causal effect (from within-tier rate noise among given doses, finger-tier hours)
- dG(finger) per U/h of a 1-h dose: L=1 -1.98, L=2 -2.23, L=3 -1.75, L=4 -1.27, L=6 -0.45, L=8 -0.14, L=10/12 ~0.
- Same across tiers (no glucose-level dependence); roughly linear in dose (no saturation seen up to ~6 U/h).
- Naive regressions including skips give a fake 'rebound' (+2 at 12h) -> artifact.

## Structural IRFs (run 22), triangular deconvolution of reduced forms on policy innovations
- Insulin (finger glucose, per U/h 1-h dose): [-1.98,-2.33,-2.01,-1.63,-1.12,-0.88,-0.59,-0.45,-0.26,-0.10,-0.08,0]; cum -11.4
- NE (map_arterial, per mcg/kg/min 1-h): [3.37,2.56,1.79,0.98,0.45,0.22,~0...]; cum ~ +9.4. Linear across rate strata.
- NE policy: titration NE_t = NE_{t-1}*mult(MAP_t)*lognormal(sd .29), mult ~1.09 (<65), 1.0 (65-75), 0.9 (>75, stop 19%/h);
  starts when arterial MAP<65 (arterial drives decisions). Cap 1.0.

## Model diagnostics (passes 1-3)
- Original dose_edit on tuning[0:100]: ins x2 -1.85, ins x1/2 +0.78, ne x2 -0.50, ne x1/2 -0.60. My estimated truth on same stay-hours:
  -13.07, +6.54, +4.75, -2.38 (population ~ -13.6, +6.8, +4.6, -2.3). Readout = pooled mean over reached stay-hours (rows only at
  recorded reading hours 13-24 of target code; reached = drug dose at h>=12 started before reading).
- Model insulin response has right time-shape but ~1/7 magnitude. NE: negative for both x2 and x1/2.
- Factual original tuning[0:100]: value NLL 1.63461 (n=4751). Model's teacher-forced slope of next glucose pred on within-tier dose:
  +0.17+-0.26 (truth -2). NE slope 0.03+-0.8 (truth ~3). => model hasn't learned causal effects at all.
- States: dose explains 14% of z posterior var at insulin bundles, ~0.3% of dOut/dH. z dose-direction also varies at reading bundles.
- Concern: amplifying full-dose pathway double counts systematic (tier-driven) dose effect absorbed in reading dynamics.

## Restart-2 findings (runs 37-47, pass 5)
- ops 1-2 = rank1 'in' on embed.numeric.weight along e2 (z_dose) and e4 (z_rate), alpha=4 (cols x5). Pass 4: ins x2 -2.51, half +1.66, ne x2 +0.14, half -0.02.
- Per 'since' (hrs since first edited dose), model/true ratio ~0.15 at all lags for insulin x2 (lag1 -0.43 vs -2.90; lag12 -3.37 vs -19.55): time shape OK, per-step gain too small.
- NE: original's negative-for-both comes from cap stays (rate 1.0); at cap half gives -0.7 (true -1.69 lag1); x2 at cap -0.5..-1.0 (OOD: rate 2.0 -> bin16, z 2.6 > train max 0.92). ops1-2 flattened NE responses to ~0.
- z posterior: at insulin bundles only z dim 26 varies (var 2.28); z26 ~ 0.65*innov + 1.37*exp_rate - 0.31*g/50; z26 also varies at reading bundles (sd 1.6) -> not selective.
- Factual-cost estimate (Gaussian, sigma_gf=14.5): lag-1 effect -k*(dose-d0): d0=0,k=1.98 -> +0.0072/reading; d0=1.5 (z-score centre) -> +0.0023. Centering at median makes raw-dose amplification affordable.
- Original model residual ~ innovation slope -1.37 at finger readings after insulin (model under-uses dose).
- ids100 = tuning stays 2020000000..2020000099.

## Restart-3 (runs 58-60)
- Pass 6 = embed.numeric z_dose/z_rate cols x5: ins x2 -2.54, half +1.69, ne x2 +0.13, half +0.06 (orig -1.85,+0.78,-0.50,-0.60).
- Pass 7 = 30 rank1 'out' ops on embed.bins.dose/rate (15 directions each, alpha 2, i.e. bin variation x3):
  ins x2 -1.88, half -0.21 (wrong sign), ne x2 -0.33, half -0.84 -> FAILED, cleared.
- Run 57: dose-induced variation in out is <0.08 of natural sd in every coordinate -> no dose-selective coordinate in out.

## z26 lever (runs 61-66)
- At ALL drug bundles only z26 varies (ins 2.31, ne 2.23, both 2.18 var); other z dims ~const. Reading bundles: many dims vary, z26 var 2.48
  but z26 ~ linear fn of other z dims (R2~0.999). z26 at ins bundles ~ 1.43*ins_rate (R2 .53); at ne bundles ~ 2.41*ne_rate (R2 .38).
- Selective edit: z_in.weight <- W (I + a e26 v^T), v.z ~0 at reading bundles (ridge-10 LS, HO leak sd ~0.12 vs z26 sd 1.57),
  v.z = z26 - type mean at drug bundles. Implemented as 4 rank1 'in' ops (balanced SPD factorisation, cond ~22). v saved /work/v_z26.npy.
- E1: alpha=4 (z26 deviation x5 at drug bundles). Testing in pass 8.
- PASS 8 (E1 z26 x5 at drug bundles via z_in): readouts unchanged (-1.83,+0.76,-0.50,-0.60). z->state pathway is NOT the dose carrier. Cleared.

## Restart-4 (runs 74-78, pass 9)
- Pass 9: numeric z_dose/z_rate cols x20: ins x2 -2.11, half +1.50, ne x2 +0.15, half +0.00 -> input-side scaling SATURATES (x5 was -2.54). Dead end.
- Idea: remap z-space in z_in (W<-W M, M=I+k g p^T): dose signal p (drug bundles) -> glucose-surprise direction g (reading bundles). Checking if g exists.

## Restart-5 (runs 91-97)
- Slots of h are identical copies (rel diff 1e-4); h norm ~842 (accumulator), out=LN(h) norm ~13.9.
- Pass-5 (numeric x5) diffs: per-step injection along dose dirs (ins sd 4.5, ne 21.6) << natural step sd along same dirs at reading bundles (~100-110) -> NO selective direction in h; state-space edits (b_in/c_out) ruled out.
- u_k is LN'd (encode_action.norm) and re-LN'd (prelude norm1) -> input scaling only rotates token -> saturates.
- z at reading bundles encodes surprises: glucose_finger dir g (dims 13,10,21,30; 0.33 per sd), map_arterial m (dims 12,13); saved /work/dir_g.npy, dir_m.npy.
- z route blocker: z26 mixes insulin & NE (cross-effects would cost huge factual NLL). Testing z-token potency first (pass 10: z_in x4 along g and m, factual both).

## Restart-6
- Pass 10 (z_in x4 along g,m): factual NLL 1.63461->1.63462, mean|dpred| gf 0.002 -> z token impotent. z route dead.
- Run109: best h-dir generalized SNR (dose-noise imprint at ins bundles vs reading-step cov) = 0.17 -> h-space amplification non-selective.
- Pass 11 plan: scale prelude.attn.out x3 (only potent drug path is prelude attention to u token).
- Pass 11/12: prelude.attn.out x3: ins x2 -2.33, half +0.89, ne x2 -0.78, half -0.98; factual NLL 1.6516 (+0.017; gf +0.027, gl +0.063) -> readings also use prelude attn; not selective. Rejected.
- Run113: dose-noise imprint in OUT at glucose query: out-of-sample generalized SNR 0.07-0.22 -> head-side/state remaps add 2-4x more noise than signal. Only selective entry = action token path.

## Restart-7 (runs 115-117)
- Cleared stale prelude.attn.out op (ops empty).
- Prelude attn: q (192x192), kv (384x192), out. encode_action: mlp1 (768), mlp2 (768), size, norm (only touches action token -> selective).
- Keep outputs short (classifier blocked long output).
- Pass 13: encode_action.norm.bias=0 -> readouts unchanged. Pass 14: action token u=0 -> all readouts exactly 0 (u is sole carrier).
- Pass 15 factual u=0: NLL 1.63461->1.63820 (+0.0036); |dpred| gf 1.12, map 0.32; per-U/h lag coef ~0.02-0.16 (truth ~2) -> u potency ~1/13.
- Pass 16/17: steer encode_action.norm.bias along mean-h dir (alpha +300/-302): ins x2 -0.92/-1.39 -> weaker both signs. Dropped.
- Next: per-head scaling of prelude attn.out columns (48 per head) to find an action-specific head.
- Passes 18-21 head scaling x4 (prelude attn.out cols): h0 -2.53/+1.67/-0.36/-0.97; h1 -1.50/+0.45/-0.37/-0.52; h2 -1.48/+0.54/-0.77/-0.71; h3 -2.05/+0.60/-0.98/-1.11. Nonlinear (base shifts). h0 best.
- Pass 22: head0 x4 factual 1.64132 (+0.0067) -> too costly. Next: sharpen head0 attention (q rows 0-47 x3).

## Restart-8
- Pass 23: prelude q rows 0-47 x3 (sharpen head0): -1.83/+0.75/-0.51 -> no effect. Cleared ops.

## Restart-9 (runs 155-167)
- Found ops = head0 x4 again (48 basis 'in' ops alpha 3 on prelude attn.out cols 0-47); purpose lost. Cleared.
- Head0 x4 factual cost (vs fwd_2 original) spread over all events & time-since-drug bins (incl >12h): gf +.006, gl +.018, art +.011;
  pred shifts: gl mean +0.80 sd 1.9, gf sd 1.9, MAP +0.3 bias; corr with residual ~0 -> non-selective noise+bias. Head scaling = dead end.
- normalize(h): 1 dominant dir (184.8/192 uncentred); variation 21 dims (99%), 40 (99.9%). kv/q/out have no bias; u-token at reading
  bundles = LN1(0) = norm1.bias. Selective LN1-space edit would need u-token geometry (unobservable) + would amplify drug-presence offset.
- Plan: modest combo. Step 1: factual cost of numeric z_dose/z_rate x5 (never measured).

- Pass 24: numeric x5 factual cost +0.00156 (gf +.0017, gl +.0046, art +.0028). Lag-1 within-tier slope change only -0.14/U/h (data ~-2).
- Pass 25: numeric x5 + head0 x2.5 (50 ops): ins x2 -2.905, half +2.286, ne x2 +0.313, ne half -0.242 (orig -1.847,+0.777,-0.498,-0.600).
  Pass 26 = factual cost of this combo.

## Restart-10 (runs 175-188)
- Pass 26: combo numeric x5 + head0 x2.5 factual 1.63969 (+0.0051) -> OVER budget. Rejected.
- Numeric feats [z_value,has_value,z_dose,has_dose,z_rate,has_rate]; z_dose==z_rate and dose bin==rate bin for both drugs (NE dose=4.8*rate).
- NE never in bin 16 in train (cap 1.0); x2 sends 54% NE atoms to bin16. Insulin uses bin16 (0.7%, max 14.5).
- Orig NE x2 by band: <=.5 model -.33 (true +.73), .5-1 +.28 (4.57), cap -.70 (6.88). Insulin ratio ~0.14 both bands.
- Pass 27: bins row16:=row12 (dose&rate): -1.87/+0.78/-0.42/-0.60 -> bins barely matter. Cleared.
- Heads read z from PRIOR (not posterior). LN1 bounds u-token deviation -> input-side gains capped (~-2.6 ins x2).

## Restart-11 (runs 190-195)
- Found ops = numeric x5 (ops1-2) + head0 x2 (48 basis 'in' ops alpha 1.0, cols 0-47) = 'N5+H2', untested. Linear cost model: N5 .0016 + head (c-1)*.0022 -> ~.0038 predicted.
- Pass 28: N5+H2 factual +0.00368, stay-bootstrap SE .0014 (N5 .00156+-.0011, N5+H2.5 .00507+-.00165) -> head0 boost too risky.
- Bands (stay mean ins rate h>=12: <=1.5 [37 stays], 1.5-2.5 [17], 2.5-3.5 [6]): truth x2 [-6.3,-16.0,-24.7], half [3.2,8.0,12.4].
  orig x2 [-.82,-2.56,-2.70] half [-.11,1.11,3.35]; N5 x2 [-2.19,-2.70,-.98] half [.23,2.61,4.50]; head0x4 x2 [-.96,-3.68,-3.88] half [-.02,2.15,6.74].
  N5 saturates doubled high doses (z=2x-1.5 -> 5z~22); head0 boosts mid/high. u-token total swing ~7 -> input-side cap.
- DECISION: final = N5 only (ops 1-2). Undid head0 ops (run 215). Next: verify N5 factual on tuning 100-199 (both).
- Pass 29: N5 factual on tuning 100-199: +0.00201 (1.62044->1.62245). Pooled 200 stays ~+0.0018.
- Pass 30: dose_edit both on 100-199: orig -1.879/+0.868/-0.312/-0.647; N5 -2.328/+1.819/+0.179/-0.239; truth {'insulin_x2': np.float64(-13.64), 'insulin_half': np.float64(6.82), 'ne_x2': np.float64(4.49), 'ne_half': np.float64(-2.24)}

## Restart-12 (runs 219-227)
- Pass 31 = N5 factual on tuning 200-299: 1.61959->1.62151 (+0.0019; gf +.0046, gl +.0133, art -.0033, cuff 0).
  Pooled 300 tuning stays: +0.00212 +- 0.00081 (run 218).
- Re-read ARCHITECTURE: drug info enters ONLY via u_k (DeepSets+LN) -> prelude attention (LN1) -> prelude_norm -> E.
  Heads use own-code register; rollout samples only glucose/MAP values (other events as recorded).
  Drug-specific handles = constant code/unit rows only -> can only rotate u, not raise potency. No new lever.
- FINAL DECISION: submit N5 (ops 1-2). Expected readouts ~ ins x2 -2.4, half +1.75, ne x2 +0.15, ne half -0.1.
