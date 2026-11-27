# Notebook

## Run 1-2: read ARCHITECTURE.md, README. Doses enter via action atoms -> encode_action -> u_k -> z_post and prelude token.

## Runs 2-13: data structure
- All stays 25.17h; readings on the hour (-1..24), drug records at h+5min (0..24), each runs 1h.
- Norepi dose = 4.8*rate (fixed weight), rate capped at 1.0. Insulin dose = rate. Rates in 0.5 steps.
- INSULIN POLICY (sliding scale on recorded glucose at same hour, finger else lab): g<150 none; 150-180 base 1; 180-220 base 2;
  220-260 base 3; 260-300 base 4; >300 ~6. rate = round0.5(base*exp(eps)), eps sd~0.31.
  Thresholds sharp at recorded value (180 exactly = 50/50) -> policy uses measured value, not latent.
- eps is exogenous: corr with eps_prev, past dg, lab-finger, MAP, norepi, hour all ~0; no stay-level component.
  => within-band dose variation is randomised; regression of future glucose on rate | g_now identifies causal effect.
- finger vs lab: no bias, sd diff 9.1. cuff = art + 3 (sd 6.7). hourly finger diff sd 14.3, lag1 autocorr of diffs -0.41.

## Runs 15-17: insulin effect (g-estimation, linear SNMM, instrument = rate residual given integer glucose)
- per-unit (1 u/h for 1h) effect on finger glucose at lag j (later doses fixed):
  [-1.96,-2.36,-2.22,-1.69,-1.33,-1.02,-0.63,-0.44,-0.47,-0.29,-0.21,-0.09], cum -12.7 over 12 lags.
- Same at lags 1-4 across glucose bands (150-180, 180-220, >220) -> additive, ~linear, not prop. to G.
- Total effect incl. policy feedback (OLS with g fixed effects) peaks lag 2 at -2.25, decays faster.

## Runs 18-23: norepinephrine
- NE POLICY: rate_t = clip((rate_{t-1} + D(MAP_t)) * exp(eta), 0.01, 1.0); eta sd~0.3 multiplicative.
  D: MAP<55 +0.08; 55-65 +0.04; 65-75 0; >75 -0.04/-0.05. Arterial used if present, else cuff.
- residual (rate - cell mean given prev rate x MAP cat) exogenous: all corr ~0.
- g-estimation NE -> MAP art per 1 mcg/kg/min: [3.18,2.81,1.58,0.80,0.57,0.03,0.14,...~0], cum +8.6 (cuff +8.1).

## Fwd 1 (dose_edit, original, tuning 100 stays in /work/sel_tuning_A.json): rows = recorded glucose_finger / map_arterial hours 13-24;
  reached = after first edited dose record (>=12). Readout = pooled mean over reached rows.
  original model: ins_x2 -0.545, ins_half +0.003, ne_x2 +0.193, ne_half -0.173.
  g-est truth on same rows (linear superposition of recorded doses from 12): ins_x2 -14.3, ins_half +7.15, ne_x2 +4.11, ne_half -2.05.
  => model nearly ignores doses. Need to inject causal dose effect.

## Runs 38-42: decoding of doses from original states (fwd_2, 100 train stays, ridge CV by stay)
- g_last R2 0.98; ins lag1-3 R2 0.71/0.63/0.57; ne lag1-3 0.60/0.56/0.54 (mostly via glucose/MAP).
- RESIDUAL doses (exogenous part): insres1 R2 0.25 (all), 0.03 (out/h only); neres1 0.15 / ~0.
=> state does not carry dose memory; head-only readout insufficient. Need core-level memory injection.
- Principle: edited prediction = m0 + sum_j beta_j*(dose - policy_mean(g)) (g-formula; centering keeps factual NLL).

## Runs 43-67 (recalled after restart): building a local torch reimplementation of the forward pass (lin, mlp, lnp, gapf, enc, encA, emb_visit2 ...) checked against fwd_2 states. Best variant so far: gelu_tanh + 'fuse', rel err 0.0865 -> not exact yet.
## Run 68: re-read ARCHITECTURE.md.

## Runs 70-91: rebuild check vs fwd_2 states
- states x: [:192]=out, [192:960]=h slot-major (out==LN(mean h) exact, eps 1e-5), [960:992]=z post mean; rows align 1:1 with bundles_B.
- embed tables: idx0 = padding (zero) for ekind/categorical/text/bins; untrained rows (~0.27 norm): action, discharged_to, route, status, text, kq 1-3 -> always 0 (kq=0 trained). unit 2-5 and rate_unit 2-3 used.
- z_post([ctx,e,u]) order confirmed; GELU best everywhere; first 32 = mean. Numeric order [zv,hv,zd,hd,zr,hr] confirmed.
- BUT z mismatch rel err ~0.6 (corr per dim 0.65-0.93) with ctx = out + LN(mean h_flow) (no gap). gap term as documented makes it worse. Visit-only bundles err 0.19. Not resolved.

## Runs 94-101: EXACT REIMPLEMENTATION FOUND (core step matches states to ~1e-6)
- atom vec = code + kind + ekind + kq[0] + [unit, rate_unit, bins, numeric only for kind 0] + numeric@[zv,hv,zd,hd,zr,hr] + numeric.bias.
  unit idx: glucose_finger 0, glucose_lab 5, map_art/cuff 2, insulin 3, norepi 4. rate_unit: insulin 2, norepi 3. bins side 'left'.
- END atoms (kind 2) of drugs go to the WORLD encoder (e), with NO units/attrs. Action encoder gets only kind-0 drug atoms.
- empty set -> encoder applied to empty set (LN(mlp2(size(0)))), both for e and u (NOT zero).
- ctx = out + LN_out(mean_slots(h*exp(-dt*exp(flow.a)))) + gap(log(dt_s)/10); z = z_post([ctx,e,u])[:32]. GELU (erf) everywhere.
- prelude: tokens [hf+slots(4), e, u, z_in(z)], pre-norm layer (attn norm1, ffn norm3), E = prelude_norm(slot outputs).
- blocks: xb = hf+E; per layer: +selfattn(norm1), +cross(norm2(x) -> M + age_emb(log1p(age_s))) [age added to memory BEFORE kv proj; M INCLUDES current e], +ffn(norm3).
- h_new = abar*hf + sp(delta_in)*b_in(E) + (xb-(hf+E)); y = coda(c_out(h_new)) (attn norm1, ffn norm3); out = LN_out(mean(y)).
- attention: kv proj -> k=first 192, v=last 192; 4 heads of 48; scale 1/sqrt(48); no biases.

## Run 108 (after restart): HEAD structure (ARCHITECTURE.md lines 54-79)
- view = out + out_norm(mean_slots(h after flow)) + gap(log dt) + z_head(z); z drawn from PRIOR 4x at eval, probs averaged.
- q = query_norm(view + type_emb[type]); with_code = q + embed.code[code] + last_numeric([has_last, log1p(age)/10, z_last_value, z_last_rate])
  + embed.bins.value[last value bin] + embed.bins.rate[last rate bin] + last_status. ('last' = most recent earlier value of same code)
- value logits = attr_heads.value_bin(with_code) + embed_transition.value_bin[code*17+last bin]; softmax 17 (0=no value, 1..16 fine bins);
  predicted mean = sum p(b)*centre[b].
- dose edit: edited drug atoms get rate&dose scaled (z-scores and bins change). readout = mean over 20 paths (common RNG), hours after drug ran.
- inspect.getsource fails for kernel functions -> use signatures/tests.

## Runs 119-121: FRESH local implementation (float64) verified teacher-forced: rel err median 1e-7 (out,h), 3e-7 (z), max 6e-3.
- kernel fns: avec(atom), senc(vecs,name), enc2(atoms)->(e,u), cstep(out,h,dt_s,e,u,Mv,Mage)->(out,y,z,E,hf), tf_check(sid).
- memory = e of last <=128 bundles incl current, ages in seconds; z-score clip +-10; fine bin = 1 + #edges < x.
- states npz keys: x, stay_id, hour, has_insulin, has_norepinephrine, n_atoms.

## Runs 122-128 (after restart 2)
- Run 122: initial state = zeros (h, out) with first dt = 1 s reproduces first state row (err 9e-8). Local core exact.
- ARCHITECTURE: a_bar fixed per dim (params only); new h = y = coda(c_out(h_new)); readout = head predicted MEAN (sum p(b)*centre), edited - factual, 20 paths.
- Shapes: value head = Linear192-GELU-Linear17; z_post 576->192->64 (only mean 32 used in core); FFNs 768; b_in, c_out 192x192 no bias; last_numeric (192,8) (8 inputs, layout unknown); type_emb 15 rows; last_status 2 rows.
- fwd_3 = factual original on the 100 train stays of fwd_2 (selB): NLL/value 1.4364 (gf 1.538, gl 1.304, map_art 1.331, cuff 1.597). Columns: stay_id,hour,event,observed,predicted_mean,nll.
- Kernel vars: bundles_B (100 stays), X_ (fwd_2 states), SIDS, W64 (float64 weights), cstep, enc2, tf_check (loop template).
- PLAN: verify head locally vs fwd_3 predicted means; then build edit = residualized dose -> glucose/MAP belief shift at drug bundles,
  calibrated so rollout effect matches g-est (ins x2 ~ -14.3, x1/2 +7.15, ne x2 +4.11, x1/2 -2.05 on tuning sel A).

## Runs 137-154 (after restart 3)
- Prelude attention at drug bundles: slots attend ~100% to u token (heads 1,2 =1.0; heads 0,3 ~0.96). At reading bundles: head0->e 0.99, heads1,3->u(empty) 0.95, head2->slots.
- |y| ~1792 (huge), abar=exp(-sp(delta_in)*exp(a_in)) ~0.36, sp(delta_in)~1, flow decay/h median 0.98.
- ins x2 at one bundle: u cos 0.97, attn_out change ~122 (|attn_out| 530), E rel change 7.5%, but y rel 0.5%, out rel 0.3%.
- HEAD VERIFIED: type_emb index 1 for readings, last_status row 0, last_numeric cols: 0=has_last, 3=log1p(age_s)/10, 6=z_last_value (others 0).
  view = out_prev + out_norm(mean(y_prev*exp(-dt*exp(flow.a)))) + gap(log(dt)/10) + z_head(z); z = mu + exp(s)*eps from z_prior(view0) (2nd half = LOG-STD).
  4-draw local NLL 1.434-1.439 vs fwd3 1.4364; rmse vs fwd3 = rmse local-vs-local (~1.8). Kernel: H (5673 rows), V0, fixed, TRr, cen, probs_from_v0().

## Restart 4 (run 157)
- PLAN: belief-injection edit. At drug bundles add to core state kappa*(rate-c_hat)*v, v = state change per unit change of last reading (finite diff, local core). Calibrate kappa with dose_edit 'both' on tuning sel A vs g-est truth; check factual NLL.
- Runs 162-178: cstep_h (hooked core, exact). Bundles: readings h:00, drugs h:05 (separate). Gate tests: dose not decodable from blocks-FFN hidden/E (R2<0.6); rates decodable from u token (R2 0.98/0.97); c_hat (E[rate|state], ridge on [out0,mean h0], R2 val .78 ins/.53 NE) decodable from prelude n3 (R2 .995/.97). Drug vs non-drug linearly separable (margin .84).
- DESIGN: stage1 = rank-1 value edit in prelude attn head1 (reads rate from n1(u), zero on u0) writes tiny signal d into slots; stage2 = 3 repurposed smallest prelude-FFN units, GELU-gated by discriminant (twin cancels offset), compute G*(rate_signal - c_hat) at drug bundles; write w_out so that dh_new = kappa*v (v = mean belief direction: dh_new per 1 mg/dL lower glucose / 1 mmHg higher MAP). Head response: inj of v moves next mean -0.30/unit (glc), +0.22/unit (MAP).

## Runs 186-212 (after restart 5): NEW DESIGN BUILT AND SAVED (edits: prelude ffn.0.weight/bias, ffn.2.weight, b_in.weight)
- mystep(d, R, dP, dhn): transparent exact core (0 err vs cstep_h); returns (out,y,z,E,hf,h_new). WCUR switch: WCUR=WE uses edited dict.
- steps_all[sid] = run_tf(bundles_B[sid], keep=True) step dicts (out0,h0,dt,e,u,Mv,Ma,res); u_empty = empty-action u.
- Slots ~identical. Prelude LN scales vary (CV .22-.31). E lives in low-dim manifold: 42 eig<1e-6 -> QUIET directions q_I,q_N (sd(qE)~4e-4 also under x2/x1/2).
- Edit: prelude FFN units 502(A_I),131(A_N),140(O) (contrib ~0): pre = 25*s + 0.5*x (s=gate ridge on n3: drug>=.87, non<=-.78; x = ridge decoders of
  (rate - c_hat) from n3 fit on factual+x2+x1/2, R2 .90 I/.81 N); output carriers c=q/gamma (mean-removed) scaled so q.E = 0.1*x.
  b_in += sum_k (t_k/0.1/softplus(delta_in) - b_in q_k) q_k^T, t = kappa*vbar (vbar = mean dh_new per +1 mg/dL glucose_finger / +1 mmHg map_art).
  c_hat = ridge(rate | [out0, mean h0]) lam 100 on drug bundles (R2 .77 I, .49 N). Parts in /work/edit_parts.pkl.
- Current kappa_I=-4.6, kappa_N=+6.1 -> factual next-reading shift: glucose -1.27 per u/h resid, MAP +1.48 per mcg/kg/min resid; cross ~0.1.
- LOCAL NLL MUST USE obs-value fine bin + lv='logstd' (matches tool: 1.434 vs 1.436, corr .96). ('target' field is NOT the bin value.)
- Edited factual dNLL (local, 100 train stays) = -0.0030 (improves!). Orig model errors regress on decoded resid: glc -1.6/u/h, MAP +2.2 -> dose signal real & ignored.
- NEXT: fwd dose_edit both on tuning sel A -> rollout gains; scale kappas to targets (ins x2 -14.3, x1/2 +7.15, ne x2 +4.11, x1/2 -2.05).

## Restart 6 (run 250+): RECONSTRUCTED (runs 213-249 were not logged!)
- fwd_4 = dose_edit both, tuning selA (100 stays): ORIGINAL ins x2 -0.545, x1/2 +0.003, ne x2 +0.193, ne x1/2 -0.173;
  EDITED (current edit, kappa_I=-4.6,kappa_N=6.1): ins x2 +0.367 (!), x1/2 -0.244, ne x2 +0.054, ne x1/2 -0.008. => EDIT FAILED IN ROLLOUT.
  Edited factual rollout does move (stay 2020000016 h13 factual 182.8 vs 178.8 orig) but x2 shift at h13 only +0.06.
- fwd_5 = states both, 13 tuning stays (factual). Runs ~230-249 compared local edited states vs tool: mismatch ~1e-3 rel (vs 1e-7 for original);
  bin side (left/right) for rate/dose does not explain it; SOLU3 (192-vec correction to u?) reduces err to 4.6e-4. Unresolved.
- No IPython history available (In=None). LOG EVERY FINDING HERE.

- RUN 258-260 ROOT CAUSE: local WE/WCUR contained an UNSAVED rank-2 edit of core.prelude.0.attn.kv.weight (rows 240-287 = v of head 1,
  sv 29462/12075, max 1928) = stage 1 (rate signal from u into slots). Decoders were fit WITH it. Tool ran without it -> factual injection
  +3..+7.7 reading-units at every insulin bundle (drift +12 by h24) and no dose response. With kv (WE): factual inj mean ~0, x2 mean -4.9
  (expected -4.2), x1/2 +0.8 (exp +2.1), noisy per bundle. Run 261: SAVED kv edit -> 5 tensors edited. Next: dose_edit both on selA.

## Runs 262-283 (restart 6)
- fwd_6 (dose_edit both selA, FULL edit incl kv, hash ee4374ba4c): ins x2 -9.43, x1/2 -1.63, ne x2 +3.21, x1/2 -0.40 (orig -0.55, 0.00, 0.19, -0.17).
  Per hour ins x2 -3.1 (h13) -> -14.1 (h24); factual path drift edited-orig: glucose +0.7 -> +4.6, MAP -1.4 -> -6.8.
- fwd_7 states both (13 tuning stays): tool edited == local WE (rel err 2e-6). W64 had the kv edit too (in-place bug) -> FIXED (W64 = true original now).
- Local single-bundle responses linear & symmetric (slopes ~0.9 x2/x1/2). Local h13 (teacher forced to 12:05): ins x2 -2.66, x1/2 +1.28; tool -3.08/+0.09.
  Tool h13 factual of EDITED model differs from local by median 2.1 mg/dL (orig 0.37) -> dose_edit rollout encodes something differently
  (not u=0, not z sampling, not END/dose drop; right-side bins for h12 atoms reproduce tool x2 at h13 exactly (-3.10) but not x1/2). UNRESOLVED.
- Without stage 1 (kv), (rate-c_hat) poorly decodable from n3 (R2 .72 I / .38 N) -> keep stage 1.
- fwd_8 factual both train selB: orig 1.43643, edited 1.43319 (-0.0032) -> constraint OK.
- Run 283: kappa scaling via b_in columns along q_I,q_N: c_I=1.5, c_N=1.3 (B_IN_c1 = c=1 version kept in kernel). Next: dose_edit selA + factual check.

## Restart 7 (runs 283+)
- fwd_9 (dose_edit edited selA, c_I=1.5, c_N=1.3, hash a56a7f6079): ins x2 -14.60, x1/2 -2.99 (WRONG SIGN; target +7.15), ne x2 +3.96, x1/2 -0.42 (target -2.05). Fit E(s)=a(s-1)+b: ins a=-7.74 b=-6.86; ne a=2.92 b=1.04 -> common-offset artifact.
  fwd_6 h13 (c=1): tool-local artifact -0.42 (x2), -1.19 (x1/2); recorded-run drift edited-orig grows ~linearly (const per-bundle bias?).
- Run 287: bundles hold kind0 (start) + kind2 (END, prev rate) atoms; my scale_atoms scaled only kind0. Local h13 vs tool (fwd_9):
  insulin_x2 V0: n=49 corr=0.406 tool=-4.62 local=-3.87
  insulin_x2 V1: n=49 corr=0.406 tool=-4.62 local=-3.87
  insulin_half V0: n=49 corr=0.429 tool=-0.04 local=1.87
  insulin_half V1: n=49 corr=0.429 tool=-0.04 local=1.87
  ne_x2 V0: n=66 corr=0.581 tool=1.20 local=1.14
  ne_x2 V1: n=66 corr=0.581 tool=1.20 local=1.14
  ne_half V0: n=66 corr=-0.276 tool=-0.50 local=-0.52
  ne_half V1: n=66 corr=-0.276 tool=-0.50 local=-0.52
- Run 287 result: END-atom scaling has NO effect on encoding (V0==V1); local-vs-tool per-stay h13 corr only .4-.6.
- Run 288 bin-side test (LL/RA=right all runs at 12:05/RE=right edited only):
  insulin_x2 LL: corr=0.406 tool=-4.62 local=-3.87 sd_t=5.16 sd_l=3.25
  insulin_x2 RA: corr=0.486 tool=-4.62 local=-4.53 sd_t=5.16 sd_l=3.28
  insulin_x2 RE: corr=0.464 tool=-4.62 local=-4.16 sd_t=5.16 sd_l=3.04
  insulin_half LL: corr=0.429 tool=-0.04 local=1.87 sd_t=6.32 sd_l=1.82
  insulin_half RA: corr=0.664 tool=-0.04 local=1.29 sd_t=6.32 sd_l=2.63
  insulin_half RE: corr=0.622 tool=-0.04 local=1.67 sd_t=6.32 sd_l=2.03
  ne_x2 LL: corr=0.581 tool=1.20 local=1.14 sd_t=4.16 sd_l=1.17
  ne_x2 RA: corr=0.491 tool=1.20 local=1.10 sd_t=4.16 sd_l=1.04
  ne_x2 RE: corr=0.573 tool=1.20 local=1.18 sd_t=4.16 sd_l=1.17
  ne_half LL: corr=-0.276 tool=-0.50 local=-0.52 sd_t=3.28 sd_l=0.60
  ne_half RA: corr=-0.328 tool=-0.50 local=-0.57 sd_t=3.28 sd_l=0.88
  ne_half RE: corr=-0.271 tool=-0.50 local=-0.50 sd_t=3.28 sd_l=0.62
- Run 288: bin side barely matters (RA ins_h local 1.29 vs tool -0.04); tool per-stay SD 2-5x local -> suspect z sampling (logstd).
- Run 289 (pred_mean with lv=logstd, nz=64):
  insulin_x2 logstd: corr=0.409 tool=-4.62 local=-3.82 sd_t=5.16 sd_l=3.18
  insulin_half logstd: corr=0.425 tool=-0.04 local=1.84 sd_t=6.32 sd_l=1.79
  ne_x2 logstd: corr=0.585 tool=1.20 local=1.11 sd_t=4.16 sd_l=1.15
  ne_half logstd: corr=-0.280 tool=-0.50 local=-0.51 sd_t=3.28 sd_l=0.59
- Run 289: logstd z sampling changes nothing. Run 290 bundle-merge test (S separate / M0 merged 12:00 readings+12:05 drugs at 12:00 / M5 at 12:05):
  S insulin_x2: corr=0.41 tool=-4.62 loc=-3.87 sdl=3.25 | base corr=0.982 medabs=3.26
  S insulin_half: corr=0.43 tool=-0.04 loc=1.87 sdl=1.82
  S ne_x2: corr=0.58 tool=1.20 loc=1.14 sdl=1.17 | base corr=0.982 medabs=1.82
  S ne_half: corr=-0.28 tool=-0.50 loc=-0.52 sdl=0.60
  M0 insulin_x2: corr=0.11 tool=-4.62 loc=-6.86 sdl=12.37 | base corr=0.714 medabs=14.60
  M0 insulin_half: corr=-0.14 tool=-0.04 loc=3.40 sdl=10.34
  M0 ne_x2: corr=0.20 tool=1.20 loc=2.20 sdl=5.63 | base corr=0.846 medabs=3.54
  M0 ne_half: corr=-0.10 tool=-0.50 loc=-0.96 sdl=4.35
  M5 insulin_x2: corr=0.12 tool=-4.62 loc=-6.76 sdl=12.81 | base corr=0.688 medabs=15.77
  M5 insulin_half: corr=-0.14 tool=-0.04 loc=3.77 sdl=10.50
  M5 ne_x2: corr=0.19 tool=1.20 loc=2.33 sdl=6.04 | base corr=0.835 medabs=4.01
  M5 ne_half: corr=-0.11 tool=-0.50 loc=-1.06 sdl=4.69
- Run 290: merged bundles WORSE (base mad 14-16) -> tool uses separate bundles; S base mad 3.26 glc / 1.82 MAP (orig model 0.37).
- Run 291 bin-combo test at 12:05 (rate_side,dose_side):
  LL: insulin_x2 c=0.41 m=-3.87 [base mad=3.26] insulin_half c=0.43 m=1.87 ne_x2 c=0.58 m=1.14 [base mad=1.82] ne_half c=-0.28 m=-0.52
  RR: insulin_x2 c=0.49 m=-4.53 [base mad=3.33] insulin_half c=0.66 m=1.29 ne_x2 c=0.49 m=1.10 [base mad=1.96] ne_half c=-0.33 m=-0.57
  RL: insulin_x2 c=0.40 m=-3.98 [base mad=3.42] insulin_half c=0.42 m=2.00 ne_x2 c=0.36 m=0.97 [base mad=2.00] ne_half c=-0.28 m=-0.69
  LR: insulin_x2 c=0.42 m=-3.88 [base mad=3.57] insulin_half c=0.20 m=2.15 ne_x2 c=0.47 m=1.05 [base mad=2.00] ne_half c=-0.37 m=-0.64
- Run 291: no bin combo fixes base mad (3.3-3.6). Run 292 precision-fragility test at 12:05 step:
  E64: ins: base mad_tool=3.26 |d vs64|=0.00 x2=-3.79 h=1.83; nor: base mad_tool=1.82 |d vs64|=0.00 x2=1.13 h=-0.51;
  Eb16: ins: base mad_tool=3.90 |d vs64|=2.03 x2=-4.44 h=1.25; nor: base mad_tool=1.26 |d vs64|=1.27 x2=1.35 h=-0.82;
  O64: ins: base mad_tool=3.23 |d vs64|=0.00 x2=-0.17 h=0.08; nor: base mad_tool=2.01 |d vs64|=0.00 x2=0.06 h=-0.04;
  Ob16: ins: base mad_tool=3.21 |d vs64|=0.01 x2=-0.17 h=0.08; nor: base mad_tool=2.03 |d vs64|=0.01 x2=0.06 h=-0.04;
  E16: ins: base mad_tool=3.59 |d vs64|=0.89 x2=-3.53 h=1.87; nor: base mad_tool=1.80 |d vs64|=0.14 x2=1.16 h=-0.49;
  O16: ins: base mad_tool=3.23 |d vs64|=0.00 x2=-0.17 h=0.08; nor: base mad_tool=2.01 |d vs64|=0.00 x2=0.06 h=-0.04;
- Run 293: bf16 fragility by tensor (median |d| h13 glc/MAP): kv .74/.75, ffn0 1.07/1.05, ffn2 .01, b_in .002 -> tool rollout likely low precision.
- Run 294: gate offset Bg 25->smaller (exact-equivalent rebuild of units 502,131,140). results {6.0: (np.float64(0.00022032295261651), np.float64(1.7222201278040413e-07), np.float64(6.06716418584194), np.float64(1.1281232273734131e-06)), 8.0: (np.float64(3.487309515795337e-07), np.float64(1.4379633550899888e-07), np.float64(5.433356165084568), np.float64(4.880935600588618e-12)), 10.0: (np.float64(3.0770388502648285e-07), np.float64(1.2687907216854685e-07), np.float64(4.779281375545592), np.float64(0.0)), 12.0: (np.float64(2.666768335543539e-07), np.float64(1.0996187718852107e-07), np.float64(4.125276994863453), np.float64(0.0))}; selected None.
- Run 295 crashed: I clobbered kernel global W0/W2/b0 (lin_t uses one) -> use z-prefixed names; W0=W64 restored if needed.
- Run 296: x-gain scaling (exact, twin adjusted). frag=median bf16 |dx| at drug steps: {'cur': (0.2059, 0.6839), (3, 10): (0.0, 6.9519, (0.819, 0.2252)), (4, 20): (0.0, 7.7428, (0.1592, 0.0997))}; saved None.

## Restart 8 (run 297)
- Run 296: x-gain variants not exact at non-drug steps -> nothing saved. Saved state = c_I=1.5, c_N=1.3 (fwd_9, hash a56a7f6079).
- b_in backup /work/b_in_current_c15_c13.npy; B_IN_c1 (kernel) = c=1 version.
- Decision: stop debugging tool x1/2 asymmetry (x2 matches design, x1/2 fails; cause unknown; only 4 runs left).
- Plan: factual both + dose_edit both on fresh tuning set C (/work/sel_tuning_C.json), then D; retune c only if needed; then FINAL.
- fwd_10 factual both tuning C, c_I=1.5/c_N=1.3 via b_in: orig 1.65072 edited 1.66997 (+0.0193) FAILS constraint (glc_lab +.033, map_cuff +.024, glc_f +.017, map_art +.005) -> b_in scaling damages broadly.
- Run 298: b_in -> c=1 (B_IN_c1); kappa scaling emulated only on edited ffn.2 cols: W2[:,j]=(I+M)W2[:,j], M=solve(B1,Dbin) rank2 (ok=True). Backup c=1 ffn2: /work/ffn2_c1.npy. Next: factual C, dose_edit C. Fallback: ffn2_c1 (pure c=1), then drop all.
- fwd_11 factual C, ffn2-col scaling only (c_I=1.5,c_N=1.3; b_in c=1; hash 2a36120ae6): edited 1.66966 vs orig 1.65072 (+0.0189) -> still FAILS; harm NOT from b_in collateral; from 1.5x/1.3x injection or base edit on tuning stays.
- Run 299: restored pure c=1 (ffn2 from /work/ffn2_c1.npy; b_in=B_IN_c1). Next: factual both C. If > +0.005 -> run 300 drop ALL edits.
- c=1 factual on all 400 tuning stays (fwd_13-16): +0.0073,+0.0072,+0.0089,+0.0065 (mean ~+0.0075) -> FAILS; ~79%% from map_cuff.
- Run 300 (LAST): family kI=1.00 kN=0.30 (pred dmg 0.0019 on 400 tuning). Family: ffn2 3 injection cols scaled along insulin/NE dirs of M; b_in=B_IN_c1; kv/ffn0 unchanged. Choice by attributing c=1 damage to insulin/NE events (<1h), conservative scaling, budget 0.002. Verify with factual blocks + dose_edit.
