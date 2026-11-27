# Notebook (/work/notes.md)
## Setup facts (runs 1-3)
- Arch: actions (insulin, norepi) -> embed (embed.numeric cols z_dose(2), z_rate(4); embed.bins.dose/rate shared across codes) -> encode_action -> u_k -> z_post & prelude token -> h -> out -> heads.
- Value head: softmax 17 classes; predicted mean = sum p(b)*centre[b].
- insulin rate/dose: median 1.5 iqr 1.0, edges [.5,.5,.5,1,1,1,1,1.5,1.5,2,2.5,3,3.5,4,5]; norepi rate median .46 iqr .59 edges top at 1.0 (cap?), dose = rate*4.8.
- Data: readings on integer hours, insulin at hh:05 lasting 1h. ~25h stays.

## Findings so far (runs 4-23)
- Insulin policy: base = step(G_dec) with G_dec = gf at hour t if measured else gl at t; steps 150/180/220/260/300 -> 1,2,3,4,6 u/h; rate = round0.5(base*m), m~1.04+-0.33 iid per hour, unrelated to observables (R2~0). ~3-5% omissions (mostly first triggers after gap).
- gf measurement: always (100%) at t if insulin at t-1, else 30% random. -> IPW weights 1 / 1/0.3.
- NE policy: start if map_arterial<65 (rate higher for lower MAP); titration rate_t = rate_{t-1}*f(MAP)*noise (lognormal sd~0.29), f~1.15 (<65), 1.0 (65-75), 0.85 (>=75); cap 1.0; stops when rate <~0.08.
- Cross effects negligible (ins->MAP 0.01/u/h, NE->glucose 0.47 per mcg/kg/min).
- G-estimation (IV = multiplier residual, IPW) insulin impulse response beta_k (mg/dL per u/h at lag k): 
  [-1.99,-2.39,-2.11,-1.71,-1.21,-0.92,-0.64,-0.47,-0.20,-0.13,-0.08,-0.01] (se .05-.16). Saved /work/beta_ins.json
- Fwd 1 (dose_edit original, tuning sel1 100 stays, /work/sel1.json): model ins_x2 -1.73, ins_half +0.91, ne_x2 -0.075, ne_half -0.48.
  Implied TRUE on same stay-hours (linear SNMM): ins_x2 -14.5, ins_half +7.26. Model underestimates insulin effect ~8x.

## Restart 1 (from run 39). Recovered state
- effects.json: beta_ins (lags0..12) and gam_ne (NE->MAP per unit rate, lags 0..12): [0,3.16,2.63,1.78,0.95,0.46,0.33,-0.02,0.07,0.06,-0.14,0.11,-0.01]
- Fwd 2 = states (original) on 3 tuning stays (156 bundles). Runs ~24-38 were an attempt to re-implement core step 1 (z posterior) in numpy;
  best error ~2.9 (not exact). Abandoned for now.
- Value head = MLP 192->192->17 (attr_heads.value_bin.0/.2), shared over codes; with_code = q + code emb + registers.

- Run 44: insulin lag-1 effect per u/h constant across base levels (-1.9..-2.3), Z^2 small -> insulin effect LINEAR additive. ins x2:x1/2 ~ -2:1.
- Run 46/50: NE effect per unit rate falls with rate (10.8,7.3,5.7,4.6 by quartile) but per log-rate ~constant (2.7-3.5 lag1) -> NE ~ LOG-linear (or Emax).
  WARNING: EPSN is not clean at capped rates (encodes MAP band; capping selects high noise) -> gam_ne in effects.json is BIASED. Need interior-stratum instrument.

- Run 55: clean nonparametric NE instrument (rate - E[rate|exp-rate bin x MAP band], titration rows) + Hansen J over power codings r^p:
  p=1 (LINEAR) best J=21 (df16); log rejected (J=582). gam_ne_clean (lags1-8) = [3.28, 2.73, 2.02, 1.21, 0.43, 0.07, -0.16, 0.23] ~ same as old gam_ne. NE linear additive.
- Targets on sel1 reached hours (linear SNMM): {'insulin_x2': np.float64(-13.99), 'insulin_half': np.float64(6.99), 'ne_x2': np.float64(4.85), 'ne_half': np.float64(-2.43)}

## Restart 2 (runs 58-71)
- NOTE: content filter blocked 2 outputs; keep prints short/technical.
- Fwd1 per-hour (sel1, ins_x2): model shift h13..24 = -0.61..-2.52 (mean -1.73) vs true -3.2..-17.9 (mean -14.0). Hour-13 (first, no feedback) model/dose12 = -0.45 (median -0.31) vs true lag1 -1.99 => dose->state pathway itself 4-6x too weak, accumulation even weaker.
- beta_ins indexed lag0..12 with lag0=0, lag1=-1.99 (dose at h affects reading at h+1).
- Fwd3 = states original, 100 random train stays (selS in /work/selS.json; mix ins-only 858, ne-only 942, both 503 bundles). 5192 rows.
  has_insulin includes END-only bundles (rate 0, ~9%). Frac-hour .17 bundles = 2-atom non-drug (100 of them).
  Ridge R2 at 1237 true insulin bundles: dose surprise (eps~rate-1.03*base) from z 0.00, out 0.12, h 0.31; rate z .29/out .62/h .71; base z .48/out .88/h .88.
  => z posterior does NOT encode dose surprise; state encodes base (glucose-implied dose) strongly, actual dose weakly.
- Arch facts: z_post.0 (192,576) reads [ctx,e_k,u_k]; z_in (192,32) only in state update; z_head uses prior samples. config: 4 heads, ff 4x, 1 prelude, 2 recurrent, 1 coda.
- Idea under consideration: centred edit needed (uncentred dose amplification biases factual preds by ~(c-c')E[dose|H]).

## Restart 3 (from run 77)
- Run 76: numpy out-from-h exact (err 0). flow decay tau 600s..6e7s (so slow dims exist). value-head hidden units all feed mean (no free units).

- Run 85-95: TIME ALIGN BUG fixed (float32 hours: use minutes). eps_ins (rate-1.04*base) linearly decodable (standardized ridge a=3):
  state right after ins bundle: out .50, h .59, z .38; next bundle: out .19, h .30; 2 bundles later ~.03. -> dose surprise IS in state at lag1, fades in 2 bundles (no long memory).
  Earlier 'z 0.00/out 0.12' R2s were ridge over-shrinkage artefacts (unscaled features).
- Factual baseline (fwd4, selS train 100): NLL 1.549; resid sd gf 15.0, gl 14.6, map_a 5.2, map_c 7.7.
- Plan: head-level centred readout correction (value head) + rely on model glucose persistence for later lags; tune scale with dose_edit 'both'.

## Restart 4 (runs 108-129): EXACT HEAD REPRODUCTION FOUND
- Bundles = WORLD atoms at hour + END atoms at end_hour (insulin/NE records AND admission rows), grouped by minute -> matches states rows exactly.
- view = out + out_norm(mean_slots(h*exp(-dt*exp(flow.a)))) + gap(log(dt_s)/10) + z_head(z), z~prior(ctx), ctx=view w/o z_head. GELU everywhere, LN eps 1e-5.
- q = query_norm(view + type_emb[1]) (type = akind*5+ekind; measurement WORLD = 1).
- with_code = q + embed.code[code] + last_numeric(f8) + last_numeric.bias + embed.bins.value[last bin] + embed.bins.rate[0] + last_status[0]
  f8: col0 = has_last, col3 = log1p(age_seconds)/10, col6 = z_last_value (same code, strictly earlier minute); other cols 0 for measurements.
- logits = value_bin.2(gelu(value_bin.0(with_code))) + embed_transition.value_bin[code*17+last bin]; bins: 1+searchsorted(edges, x, 'left').
- Repro vs fwd4 CSV: rms 2.03 (gluc) / 1.07 (map) = 4-draw z sampling noise floor (1.96/1.08). => offline head is exact.
- Plan: rank-1/2 update of attr_heads.value_bin.0 (W1 += eps_vec w^T) with decoder w of dose surprise from with_code, eps_vec chosen so mean shifts only for glucose (ins) / MAP (NE). Calibrate scale with dose_edit 'both' on tuning sel1.

## Restart 5 (runs 130-144) recovered
- fwd5-7 = states original on selT (300 train stays, /work/selT.json). XALL(17848,992)/STALL(stay_id,minute,has_ins,has_ne,row,prev_row,prev_min) = all bundles of 400 stays (selS+selT=stays400).
- RD/RD2: 19451 readings of those stays (stay_id,hour,event,observed,minute,prev_row=XALL row of state before reading,last_val,last_min,pred0,nll0; RD2 adds ins_k,ne_k,eps_ins_k,ins_exp_k k=1..3, res).
- HI = offline head inputs per reading (ctx,zmu,zls,reg,trans,ob,cen,ev); WC0=with_code (19451,192); Hh0/A0 hidden; P0 probs.
- Run142 lin_sep on WC: glucose vs map separable (r2 1.0); ins-present among glucose r2 .996; joint gate not linear (.78); NE among map .98.
- Helper sources NOT retrievable (inspect fails): rewrite helpers when needed.
- Plan: per drug rank-1 W1 edit: W1 += c (W1 d) w^T, b1 -= c (W1 d)(w.mu); d = WC direction moving glucose (MAP) mean but not other code; w = centred decoder of dose signal from WC; calibrate c with dose_edit both on sel1.

## Restart 6 (runs 150-165)
- Offline head exact (pred0/nll0 reproduced). WC0 = with_code at z=prior mean. wc_from(z) rebuilt (z_head, query_norm, type_emb[1], reg). One z draw changes wc rms .23.
- Factual residual slopes on insulin surprise (PT_tr eps, lags1-6, gluc rows pooled): -1.26,-1.29,-0.42,+0.37,+0.27,+0.61 (se .3) -> model misses lag1-2 effect, later lags carried by readings.
- NE: E[rate|MAP hist] via HGB (R2 .77), D_ne = rate - E. Resid slopes on D_ne lags1-3 (arterial): 3.14, 0.15, -2.2. One-step eps_ne lags: 3.37,2.65,3.15.
- Decoders from wc with z-noise aug (ridge a=1, CV): ins eps lag1 R2 .33 (slope .32), lag2 .18; NE D lag1 .15, eps_ne lag1 .10. Full state ins lag1 .63 (head cannot see slots separately).
- Plan now: hidden-space injection u (W1 += u w^T) gated by Jacobian LS so glucose-only / MAP-only; scale c/kappa; test with dose_edit both on sel1.

- Runs 166-180: gated-channel head edit. Repurpose hidden units (31,143)=insulin, (54,9)=NE (joint ablation dNLL +.0003, 4-draw).
  unit a: M*(AND gate - th) + s/lam, unit b: M*(AND gate - th); W2 cols +-lam*t (exp tilt). AND gate = family gate + presence(drug at t-1) gate (ridge on z-aug wc).
  Decoders on present rows (z-aug ridge): ins eps1 R2 .59 k .57, eps2 .24; NE D1 .19 k .25. Present-row deficits: ins -1.6,-1.33,-0.39; NE joint 3.15,.53,-2.13.
  E2 (a_ins=(-2.8,-5.5), a_ne=8.7): offline dNLL +.0063 (gf +.0117) over budget; resid slopes fixed. SAVED as current edit for dose_edit test (fwd 8).

## Restart 7 (from run 186)
- Fwd8 = dose_edit both, sel1, E2: original {'insulin_x2': -1.726, 'insulin_half': 0.909, 'ne_x2': -0.075, 'ne_half': -0.484}; edited {'insulin_x2': -7.171, 'insulin_half': 1.473, 'ne_x2': 0.459, 'ne_half': -1.918}. Targets ins_x2 -14.0, ins_half +7.0, ne_x2 +4.85, ne_half -2.43. E2 ~half of ins_x2, ins_half barely moves (asymmetric channel), NE x2 small. E2 also over NLL budget.
- Fwd9 = factual both tuning sel1, E2: orig 1.51191 edited 1.52176 (+.0099, OVER). gf +.007 gl +.003 map_a +.012 map_c +.014; leaks on no-drug gf rows (+.012).
- KEY: dose_edit rollouts run in BF16 (factual in fp32). E2's big-activation gate units (M=100, diff of gelus) break in bf16: offline bf16 emulation gives +17.8 mg/dL bias on ins-present gf rows (fp32 -0.6) = the rollout drift (+14..+56 factual arm). Fwd8 readouts are artefacts.
- RULE: any edit must be bf16-robust (O(1) activations, no cancellation). Test offline with head_mean2(..., torch.bfloat16).
- E3-ins (runs 202-207): units (31,143) pair a=k(G-1.5)+s, b=k(G-1.5), k=6, lam=4, tilt t=cen_gf/100, all values pre-rounded to bf16. G = glucose-family gate + insulin-presence(ins_1>0) gate (ridge a=1, z-aug; clean margins). s from eps_ins_1 decoder ridge a=0.1 (cvR2 .59, slope .569, |w| 45), rescaled 1/k, C=-3.32/u/h (sum of factual deficits lag1-3: -1.6,-1.33,-0.39). NE untouched (decoders R2<.2 too costly).
- Fwd10 = dose_edit both sel1 with E3-ins (C=-3.32): ins_x2 -16.55 (tgt -13.83), ins_half +7.86 (tgt +6.92), NE unchanged. Factual-arm bias h13 +0.8 (bf16 fix works). Profile: early & late overshoot, mid ok; model persistence > true decay. Linear calibration -> C ~ -2.8.
- Run 210: NE info weak even in full state (R2 D_ne_1 full .30, wc .17; eps_ne_1 .20/.11; rate .57/.46) -> NE head channel limited. E3b = E3-ins with C=-2.79 saved (/work/E3b.npz).

## Restart 8 (from run 212)
- Run 211: E3b (C=-2.79) offline dNLL all +.0004, glu&pres +.0098, map -.0036, held-out +.0016; bf16 bias -.02.
- Keep replies/outputs short (filter).
- Runs 216-223 (NE): NE info at MAP_a rows weak (D_ne_1 R2 full .32, wc .18; prev-bundle-NE no better). Decoder must be fit with zero targets on off-gate rows (om=.1) else extrapolates (dAll +.37!). cvR2 .22 k .194.
  Unbiased-decoder noise var = C^2 VarD/R2 -> affordable C~2 only. bf16 rounding noise of gate rows matters: use lam=1 (noise/sig .07).
  NE channel: units (124,142) (cheap: ablation cost <0), AND gate fam(map_arterial)+pres(ne_1>0) ridge a=1 z-aug, kap 6, tilt tA=(cenA-75)/10, Vref 20.7.
  Cost vs E3b: C1.5 +.0003, C2 +.0018 (heldout +.0012), C2.5 +.0032. Slope ~C, bf16 bias ~0.
- E4 = E3b + NE C=2 lam1 saved (/work/E4.npz, current edit). Next: dose_edit both sel1.
- Fwd11 (sel1, E4): ins_x2 -13.93 (tgt -13.99), ins_half +7.03 (+6.99), ne_x2 +1.03 (+4.85), ne_half -2.15 (-2.43). Insulin DONE.
- WHY ne_x2 weak: NE rates capped at 1.0 in records (bin edges ..0.87,1.0,1.0,1.0,1.0; dose=4.8*rate, z_dose=z_rate). x2 sends 47% of NE atoms to rate/dose bin 16 (never seen for NE; bins tables shared with insulin, insulin uses bin16 0.7%, 13% under x2) and z up to 2.6 (train max .92). Half stays in range -> works.
- PLAN: NE-specific fix in encode_action.mlp1 (only ins/NE atoms pass there): gate unit(s) on NE-code AND bin16 (directions orthogonalized vs other embeddings), add correction in mlp1 output (cap at r=1 repr or extrapolate). Zero factual cost. First verify atom_vec/set_enc (helpers exist, unverified) via z_post at action-only bundles.
- Run 236 VERIFIED encoder: XALL row = STALL.row (NOT index); prev_row = XALL row. NE START atom = code[9]+kind[0]+ekind[2]+bins.rate[b]+bins.dose[b]+bins.value[0]+numeric@[0,0,zd,1,zr,1]+bias+unit[4]+rate_unit[3]; u_k=LN(mlp2(sum mlp1/sqrt n + size(log(1+n)))); e_k(action-only)=empty-set enc (size(0)); z_post([ctx,e,u]) matches XALL z to 5e-4.

## Restart 9 (runs 239-245)
- fwd11 per-stay-hour regression (lag = h-floor(t) reproduces true_shift_ne; gam_ne_clean padded to 13): ne_x2 target 4.87 = in-range part .90 + OUT-OF-RANGE part 3.97 (doubled rate>1).
  model shift ~ -.58 + 1.54*tin + 0.06*tout  -> out-of-range NE atoms give ~ZERO response. half: shift ~ 2.26+1.81*tgt.
- mlp1 = Linear(192,768)-GELU-Linear(768,192), no pre-norm. Plan: gated NE&bin16 correction units in encode_action.mlp1 (zero factual cost).

- Run 256-266: encode_action.mlp1 gates: wGf (NE START & bin16 -> +1, others <=-1), wGp (any NE START), wZ (reads z_rate) exact, orthogonal to numeric/unit/kind embs. Cheap units 674,411,314,524 (ablation max|du|<=.004).
- u(r) for NE is BIN-DOMINATED (step fn), out-of-range moves off-manifold.
- PROBE fwd12 (states both, train selP=/work/selP.json, unit674 injection q*2.5*v1 on all NE START): footprint at MAP_a lag1 93% one direction but reader SNR~1 (natural var); E4 head reads it as MAP -0.4 (wrong sign) -> abandoned new head channel.
- PLAN: bin16 NE START mlp1 output := m(r=1.0) + beta*(r'-1)*dm_dose (in-range slope), 4 gated units; calibrate beta with dose_edit.

- Run 273-276: atom formulas: NE START = old formula + kq[0]; insulin START: code8, unit3, rate_unit2, kq0 (z_post exact). END atoms unresolved -> gate orthogonal to ALL kq/categorical rows (gate_f(-1,-4,-3)).
- E5 = E4 + encode_action.mlp1 units (674,411,314,524): bin16 NE START -> m(1.0)+beta*(r'-1)*dm (beta=1); /work/E5enc.npz.

## Restart 10 (from run 277)
- E5 (beta=1) saved; run276: factual NE du .0575, bin16 u err [.67,.4,.44]. Next: fwd13 dose_edit both sel1 with E5.
- Runs 279-282: beta scaling saturates (LN: |du| 2.6->3.9 max); u(r) two-group (0.1-0.5 vs 0.6-1.0, |du|=19.4); great-circle extrapolation gain <=1.07 -> NO in-range-readable repr for >cap NE. Keep E5.
- Fwd14 sel2 (E5): {'insulin_x2': {'model': np.float64(-13.4), 'tgt': np.float64(-13.16), 'err': np.float64(-0.24), 'se': 0.72}, 'insulin_half': {'model': np.float64(6.12), 'tgt': np.float64(6.58), 'err': np.float64(-0.46), 'se': 0.76}, 'ne_x2': {'model': np.float64(2.28), 'tgt': np.float64(3.78), 'err': np.float64(-1.5), 'se': 1.82}, 'ne_half': {'model': np.float64(-2.62), 'tgt': np.float64(-1.89), 'err': np.float64(-0.73), 'se': 0.95}}

## Restart 11 (runs 287-)
- fwd15/16 = factual both sel1 (E5 and E4): edited 1.52200 vs 1.51191 = +.0101 OVER BUDGET. gf +.0117 (shift~resid slope .22), map_a +.0193 (slope .03 = pure noise), gl +.0013, map_c -.0015. Offline cost estimates were ~4x too optimistic (decoders do not generalize to tuning stays).
- Encoder (E5) factual effect negligible (15 vs 16 identical). fwd13 = E5 dose_edit sel1: ins -13.91/+7.08, ne_x2 1.35, ne_half -2.15.
- DECISION: drop NE head channel; insulin channel W2 cols scaled f; keep E5 encoder. E6 = E3b(W2 ins cols*0.8, bf16) + E5enc, saved (/work/E6h.npz). Expected factual ~+.002 (cost(f)~.0069f^2-.0030f overall), ins readouts ~ orig + f*(E4-orig).
- fwd17 factual sel1 E6 (ins f.8, no NE head, E5 enc): +.00088 (gf +.0063, gl -.0003, map_a -.0039, map_c +.0007). cost_ins(f)~.0072f^2-.0032f overall; NE head alone ~+.023/map_a row at C=2 (~.0078 overall, ~f^2).
- E7 = E6 + NE units from E4 with W2 cols *0.4 (est cost +.0012). /work/E7h.npz. Tuning ids 2020000000-399; rest-of-sel1 chunks in /work/selR.json (R1,R2,R3).
- E7 factual: sel1 +.0020, R1 -.0016, R2 +.0004, R3 -.0018 (pooled ~-.0002). Projection model (gluc c2/c1 on sel1 f.8&1; map quad on sel1 fne 0/.4/1) -> E8 = ins f=0.95, NE f=0.50, proj pooled 0.0010 max set 0.0040. /work/E8h.npz saved as current.
- E8 (ins .95, NE .5) proj pooled .0010 max .0040 -> replaced by E9 = ins f=1.0, NE f=0.4 (+E5 enc), proj pooled .0009, max set sel1 .0039; proj readouts sel1 ins -13.9/+7.1, NE x2 ~.5, half ~-1.15. /work/E9h.npz = current. Next: factual sel1, dose_edit sel1/sel2.
- FINAL CHECKS E9 (ins f1.0, NE f.4, E5 enc; edit hash 0595dc8731):
  factual both: sel1 +.0039 (fwd22), R1 -.0006 (fwd23), R2 +.0016 (fwd24), R3 -.0011 (fwd25); pooled 400 tuning stays ~ +.0010. Map rows improve on R sets (repurposed units/b2), gluc cost = insulin channel.
  dose_edit both sel1 (fwd26): ins_x2 -13.82 (tgt -13.99), ins_half +7.03 (+6.99), ne_x2 +0.70 (+4.85), ne_half -1.14 (-2.43). Orig -1.73/.91/-.08/-.48.
  Higher NE gain rejected: f_ne .7 -> pooled ~.0025 but drug-heavy sel1 ~.0066 (over). E9 = SUBMISSION.
