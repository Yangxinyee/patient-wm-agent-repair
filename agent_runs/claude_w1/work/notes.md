# Notebook

## Run 1-2: orientation
- Read ARCHITECTURE.md and README. Dose enters via encode_action (u_k), embed.numeric (z_dose,z_rate), embed.bins.dose/rate.

## Runs 3-10: data structure & dosing policy (train.csv)
- All stays 25.17h. Readings at integer hours (-1..24); infusions at h+0.0833, each 1h. Insulin dose==rate; NE dose=4.8*rate.
- Glucose finger/lab intermittent; map_arterial hourly in ~48% stays (art line); map_cuff ~60% hours.
- INSULIN POLICY (uses the reading taken that hour, finger; lab if no finger): ON if G>160 (always); 140-160 depends on prev on (0.68 vs 0.23); <140 ~off.
  Base rate steps on recorded finger value: ~1.05 (<180), ~2.08 (180-220), ~3.15 (220-260?), higher above; sharp jump AT the recorded value
  (at exactly 180 mean 1.56 -> threshold on unrounded reading). Multiplicative noise, rounded to 0.5 (sd ~0.37 at base1, ~0.65 at base2).
- NE POLICY: start if MAP<65 (art if available else cuff); titrate: MAP<65 up (log ratio +0.07..0.15), 65-75 hold, >75 down (-0.2..-0.28);
  log-ratio noise sd ~0.29; cap 1.0. Thresholds sharp at 65/75.
- IDEA: dose noise given recorded reading = natural experiment; thresholds (180,220; 65,75) allow regression discontinuity (RD) check vs hidden confounding.

## Runs 11-23: causal effect estimates (train.csv), using exogenous dose noise
- Insulin noise log(rate/base) iid: corr with prev noise 0.000, with G/MAP/NE/h ~0.01. NE innovation (log ratio resid) corr prev -0.014, MAP -0.04.
- Measurement: finger vs lab same hour: no bias, sd diff ~5.5% of G (multiplicative). cuff - arterial = +3.0 mmHg, sd 6.7.
- No cross effects: NE->glucose ~0, insulin->MAP ~0.
- Insulin -> glucose: local projection total effect per +1 u/h shock: k=1..6: -2.08,-2.29,-1.87,-1.37,-0.76,-0.45 (->0 by 12).
  Linear in dose (shock^2 ~0). Mildly larger at high G: -1.83 (G~164), -2.07 (196), -2.21 (234), -2.43 (272) at k=1.
  Feedback on later doses small: -0.06 u/h. Deconvolved fixed-dose impulse response h_ins(k):
  -2.08,-2.41,-2.16,-1.79,-1.26,-0.95,-0.77,-0.63,-0.55,-0.43,-0.33,-0.30 (cum -13.7 at 12h)  [tail uncertain]
- NE -> MAP_arterial: total per +1 mcg/kg/min innovation: 3.4,5.1,5.4,5.1,4.4,3.8,...; dose persistence 0.73,0.56,0.44,...
  Deconvolved h_ne(k): 3.40,2.60,1.62,0.94,0.41,0.33,0.0,... (cum ~9.3). Linear in rate (k=1 effect 3.3-3.6 across rate levels).
- Model-side caution: NE rate>1.0 / dose>4.8 -> bins 13-16 never seen in training (x2 edit hits them).

## Run 24-29 + fwd 1 (dose_edit original, tuning setA = /work/setA.json, 100 stays)
- RD at insulin thresholds 180/220 agrees with noise-based estimates (-1.9..-2.6 per u/h at k=1) -> no hidden confounding.
- dose_edit rows = stay-hours 13-24 where the record HAS the outcome reading (finger glucose / map_arterial); reached = hour > first dose hour>=12.
  mean_shift = POOLED mean over reached stay-hours.
- ORIGINAL model on setA: ins x2 -1.448, ins x1/2 +0.824, ne x2 +0.152, ne x1/2 -0.411.
- My estimated TRUTH on same rows (deconvolved h): ins x2 -14.17, ins x1/2 +7.09, ne x2 +4.82, ne x1/2 -2.41.
  => model's effects ~10x (insulin) and ~30x (NE) too small.

## Runs 30-41 (after restart)
- fwd_2 = factual ORIGINAL on tuning setA: value NLL/reading 1.51903 (finger 1.531, lab 1.261, art 1.349, cuff 1.746). Baseline for +0.005 limit.
- fwd_3 = states ORIGINAL on tuning setA (5075 bundle rows; layout out|h0..h3|zpost).
- Runs 30-37 tried to reimplement core locally: not matching yet (best err 1.79 on z after bundle0). Parked.
- Value head = MLP 192->192->17 (+ embed_transition[code*17+lastbin]); shared by all 4 value codes.
- PLAN: readout edit. Decode a dose-innovation feature (dose minus policy-expected dose given reading, lagged) from head input
  computed from states; repurpose one hidden unit of attr_heads.value_bin per drug to add a logit tilt (gated to glucose / MAP codes);
  calibrate gain with dose_edit on setA vs my truth (-14.17/+7.09/+4.82/-2.41). Check factual with 'both'.

## Runs 42-69: replicas
- Dose innovation decodability (tuning states): insulin nu1 R2 ~0.24 from head query q (slope 0.29); NE innovation R2 0.09 (slope 0.10).
  => pure readout edit too noisy (esp. NE). Need upstream signal.
- Original model rollout insulin x2 per hour: -0.53 (h13) .. -2.28 (h24); truth -3.2 .. -18.8. Model persistence ~0.78/h.
- HEAD replica (from states): view=out+LN_out(mean_slots(h*exp(-dt*exp(flow.a))))+gap(log(dt_s)/10)+z_head(z_prior mean); q=LN_query(view+type_emb[1]);
  with_code=q+code+last_numeric([has,log1p(age_min)/10,zlast,has,has,0,0,0])+bins.value[lastbin]; GELU; MAE 2.4 vs tool (noise floor ~1.2).
- ENCODER/z_post replica EXACT (rel err 0.008 on drug bundles): GELU everywhere; empty set -> LN(mlp2(size(0))) (not zero);
  units: mg/dL->3, mmHg->2, mg->4, units->5; rate_unit: mcg/kg/min->3, units/hour->2; drug END atoms go to e (measurement encoder), not u.
  atom = code+kind+ekind+kq[0]+unit+rate_unit+bins+numeric@[zv,hv,zd,hd,zr,hr]+bias. z_post([ctx,e,u]) ctx=out+LN_out(mean h decayed)+gap(log dt/10).
- Runs 70-74: z_post replica EXACT on all bundle types. Units: mmHg->2, units(insulin dose)->3, mg(NE dose)->4, glucose_lab mg/dL->5, glucose_finger NO unit. Bins side='left'. Use atom_e_full(4,3).

## Runs 76-83 (after restart 2): EXACT replicas rebuilt (clean code in kernel: Wo, avec, encset, bundles_of_stay, zpost_of, ctx_of, core_step)
- Bundles: group atoms by round(hour*60). Drug START atoms -> action set u; drug END atoms (no attributes) + measurements + admissions(+END at end_hour) -> e.
- Atom = code+kind+ekind+kq[0]+unit+rate_unit+bins(value,dose,rate)+numeric@[zv,hv,zd,hd,zr,hr]+bias. unit: map 2, glucose_lab 5, finger 0, insulin dose 3, NE dose 4.
  rate_unit: units/hour->2, mcg/kg/min->3 (global rumap in kernel is WRONG/stale). bins: 1+searchsorted(edges,x,'left'). z clipped [-10,10].
- Set enc: LN(mlp2(sum mlp1(a)/sqrt(n) + size(log1p n))); EMPTY set (e or u) -> LN(mlp2(size(0))) (NOT zero). GELU(erf) everywhere.
- z_post exact. CORE STEP EXACT (relerr 0.0000): h'=h*exp(-dt*exp(flow.a)); ctx; z; prelude over [h'+slots, e, u, z_in(z)] (pre-norm: norm1->attn, norm3->ffn), E=prelude_norm(slots);
  blocks 0.0,0.1 on x=h'+E: x+=attn(norm1 x); x+=cross(norm2 x, M+age) where kv=Wkv(M_e + core.age(log1p(age_seconds))) [age added BEFORE kv, both k and v], M includes CURRENT e_k, last 128;
  x+=ffn(norm3 x). hm = abar*h' + softplus(delta_in)*b_in(E) + (x-(h'+E)), abar=exp(-softplus(delta_in)*exp(a_in)); y=c_out(hm); coda: y+=attn(norm1 y); y+=ffn(norm3 y). out=out_norm(mean_slots y).
  MHA: 4 heads x48, no biases, kv rows [0:192]=K,[192:384]=V, scale 1/sqrt(48).
- Dose info enters state ONLY via u_k / z_k tokens in prelude (memory M holds e only; END atoms carry no dose).

- Run 84-91: HEAD replica approx (type_emb[1], gap(log dt_s/10), last_numeric=[has,log1p(age_min)/10,zlast,has,has,0,0,0], GELU, last_status row0 = 0): MAE ~2 vs tool (not fully exact; accept).
- FREE-RUN replica EXACT from zero init (h=0,out=0, dt0=1 s): run_stay(B) in kernel, 0.07 s/stay -> can generate exact states locally for any record (incl. counterfactual doses).
- Prelude attention: slot queries attend ~100% to the u token (head2 always; heads0,1 ~97-99% on insulin bundles); z token ~0 attention. => dose path = values of u token, same for all slots.
- PLAN now: head-level edit c*(dose_decoded_from_q - pi_hat(register last value)); decoder trained on states with RANDOMLY PERTURBED doses (breaks policy link).

## Session 3 (runs 94-99)
- samp_pert.pkl = list of tuples (code, stay, hour, prev_val, cur_val, dose_lag1, lag2, lag3, with_code[192]) from gen_samples(df,pert=True), 1500 train stays, 21456 finger + 17722 art rows.
- Perturbed-dose decodability from head input (ridge, CV): ins lag1 R2 .70 (resid sd .78 u/h), lag2 .61, lag3 .57; NE lag1 .73 (resid .21). prev value R2 1.0 (register).
- Core dynamics: abar=exp(-softplus(delta_in)exp(a_in)) ~0.36 per bundle (0.29-0.44) -> state refreshed each step; persistence via loops.
  flow.a exp(a) 1.6e-8..1.8e-3 (per s). h is low-rank: slot0 PCA sd top 155,95,58; 133/192 PCs sd<0.5. No dead channels (min ch sd 2.7). 4 slots ~identical stats.
- Key theory: correct edit = add sum_k c_k * nu_{t-k}, nu = dose - E[dose|readings] (policy), c_k = missing direct coef from residual regression of
  (reading - model mean) on lagged nu. Raw-dose amplification would bias factual preds (doses track G) -> must centre on policy.
- WARNING: gen_samples(df) treats df as ONE record -> call per stay (0.05 s/stay). Runs 105/107 invalid.

## Session 4 (runs 126-138)
- Kernel archaeology: keep/mu_all/res_all = REPLICA (z = prior mean) factual means on 6000 train stays; biased (finger res mean -3.35) and
  res R2~0.18 on x -> NOT reliable for factual-fit analysis. Dm/Rm/NUm/NUNm: (30000 train_ids order) x 26, column j = dose/innovation at hour j-1
  (lag k for reading at hour h = column h-k+1). NUA/DA are NOT aligned with setA. Ptu3/Ptr3 panels (stay_id,h) have R, Rhat, nuN (NE innovation ok).
- ARCH: z drawn from prior 4x, probabilities averaged (tool factual means are noisy); readout = predicted MEAN per hour, edited-minus-factual,
  20 paths, common random numbers. Value head: 17 classes, 0=no value, mean=sum_b p(b)*centre[b-1].
  finger centres 73,84,92,104,119,137,149,160,169,178,190,205,222,236,250,272; art centres 33,38,43,49,56,63,68,73,77,80,83,87,91,94,97,101.
  with_code = q + code + last_numeric + bins.value[lastbin] + bins.rate[last rate bin] + last_status; logits = value_bin MLP + embed_transition[code*17+lastbin].
- Replica residual ~ innovation lags (train, 80k finger / 67k art): insulin R=[-1.49,-1.53,-1.00,-0.14,.25,.22,.45,.27] (se .12);
  NE R=[2.55,3.69,3.22,2.80,2.20,1.60,1.01,.71] per mcg/kg/min (se .2); NE innovations persist in later doses (0.73,0.56,...) ->
  per-dose NE missing effect ~[2.55,1.83,0.45,0.32,0]. Rollout needs per-dose effects (dose path fixed in edit).
- Ridge decoders on samp_pert (DEC dict): CV R2 ins .69/.61/.57, NE .70/.64/.59; factual decode err sd ins .53, NE .18; slopes .9/.85/.79 (ins), .79/.76/.75 (NE).
- TOOL residuals on setA (F2): MAP res~nuN lags [3.71,1.63,1.11] (se .8-1); res~(dhat-pi) lag1 = -1.48+-.75 (WRONG sign) -> decoded-dose
  correction costs factual fit unless decoder noise is reduced/orthogonalised. Factual budget: rms(extra error) <~1.7 mg/dL glucose, <~0.55 mmHg art.
- PLAN: get tool factual means on 400 train stays (4 passes) -> exact residuals; design MVDR-type decoders (unit dose response, min factual harm);
  edit = repurposed value-head hidden unit per drug (GELU gate by code group), tilt along bin centres, compensation via embed_transition per lastbin;
  calibrate gain with dose_edit on setA vs truth (-14.17/+7.09/+4.82/-2.41); check factual with 'both'.

## Session 5 (runs 141-165)
- Policy check on setA tool residuals (F2): finger res ~ insulin innov lags [-3.13,-1.63,.6] (se .63) but ~ policy part lags [-1.21,-.41,.96]
  -> policy part ABSORBED by model; raw dose amplification would bias factual (~2.8 mg/dL rms). NE noisy (similar conclusion).
- Correction must respond to dose LEVEL (sustained x2) with compensation depending on READINGS only (not past doses), else NE x2 cancels.
- u-token injection via prelude head 2 (attends 100% to u on all bundles; decoders w_ins/w_ne on norm1(u) R2 .999, zero on empty u):
  model's own dose direction useless (ins->glucose +0.38 wrong sign; NE->glucose -2.7 cross). Consistent per-query effect needs |alpha|~1000 per
  1 mg/dL/(u/h) (LN after prelude) -> NOT viable. Relay (inject to make dose decodable at head) fails: heterogeneity ~9x slope.
- HEAD-LEVEL correction c*(decoded dose - mean of decoded dose given (code,lastbin)) evaluated on setA with tool residuals:
  insulin (finger) ridge decoders from samp_pert: lag1 slope .86 errsd .57, c=-3 -> dNLL(weighted) -0.0002 (gain); level3 slope 1.0 errsd .34.
  NE (art): errsd .19-.21 (innov sd only .079); cost ~ +0.0018 at c=3, +0.0047 at c=5 (B not significant, A=.035 solid). Budget 0.005 total.
- setA reading mix: finger 34%, lab 10%, art 26%, cuff 29%; res sd 14.9/12.0/5.2/7.9.
- kernel: head_mean returns (mu, P); globals b1,b2 were stale -> reloaded W1,b1,W2,b2,TR from original. RU0 = correct rate map.
  run_rows(stays)/run_rows_wc use gen_samples(G3[s]); H100 = 100 train stays; sweep files /work/sweep_h2.npz, sweep_dwc.npy.
- PLAN: repurpose 2 value_bin hidden units (gated by code group), tilt logits along bin centres, comp via embed_transition per (code,lastbin);
  calibrate gains with dose_edit on setA vs truth (-14.17/+7.09/+4.82/-2.41); check factual with 'both'.

## Session 6 (runs 192-199, reconstructed; runs 166-191 were not logged)
- Kernel has Xf(70738x192 head inputs, 1500 train stays), cf code id (0 finger,1 lab,2 art,3 cuff), lbf lastbin, Rf=cell-residualised Xf,
  DQf dose-direction derivs, EDR[(drug,factor)]={gi,dX,dd(4 lags),reached} replica edit responses; solve_unit -> decoders vI (glucose rows), vN (MAP rows).
- diag: vI noise 15.4/7.4, own response 1.0, leak NEx2 -0.31; vN noise 3.3/0.94, own 1.0, leak ~0. Units being checked.

## Session 7 (runs 249-265)
- DEC ridge decoders (samp_pert) DO NOT respond to replica edits on Xf (resp 0.09 vs dose chg 1.42): invalid representation (gen_samples != Xf, offset ~9). Xf+W1.. replica head exact vs nll0.
- MVDR decoders from EDR dX (unit mean response per dd123) + within-cell cov of Rf (ridge 3.6e-5): noise sd ins 1.24, NE 0.20 (per unit). Stored DEC2[('ins','mvdr',-2.8)], DEC2[('ne','mvdr',3)].
- Replica exact dNLL of tilt edit (all rows): ins c=-2.8 0.0067, NE c=3 0.0199 (replica sharper than tool: nll 2.04 vs 1.52). Resid-informed decoders worse.
- EDIT v1 SAVED (run 264): units 0 (glucose, gate gw) & 114 (MAP), pre=B*gate+k*(v.x-m), W2 col = c/medvar*centred centres/k, TR per-cell compensation. cI=-2.8 cN=3.0. replica dNLL .0268.
- Next: tool factual both + dose_edit both on setA.

## Session 8 (runs 269-)
- fwd_5/6 = factual both with EDIT v1 variants: BROKEN (edited NLL 18.3/16.3; dmu ~+95 glucose, -15 MAP). Cause: big-bias gate not invariant to q/z noise. ALL EDITS DROPPED (run 271).
- ARCH: q = query_norm(view+type_emb) (LN) -> g=1/gamma_qn gives g.q = const EXACTLY (z-proof gate). with_code = q + code_emb + last_numeric + bins + last_status.
- PLAN v2: per drug 2 units, same gate G*(g.x-g0)+B, unit A adds k*v.x, W2 cols +w/-w -> output = w*k*v.x in on-group (linear regime), 0 off-group. Moderate magnitudes. Calibrate c with tool.
- Run 273-277: LDA gates (z-orthogonal, finger vs rest lamf .01, art vs rest lamf .03) margin/sd 36/45. Dead units [114,10,0,107] (ins A,B; NE A,B).
  Decoders DECR2: orthogonal to z-subspace + last_numeric + bins.value/rate + last_status + code emb (85 dims), kap .3: ins slope .522 per-unit noise 1.53; NE slope .683 per-unit .279.
  build_v4(gI,gN,...) 2-unit difference, k=4/p99.99, tilt along centred centres /medvar, Vp-weighted exact cell centring via TR. SAVED gI=-2.0 gN=2.5:
  replica: lab/cuff untouched; finger shift mean .13 sd 3.8 KL .0153; art mean -.04 sd 1.6 KL .0203; direct/h ins2 -3.26 ins.5 1.72 ne2 3.17 ne.5 -1.46.
  Earlier failures: v1 big-bias gate (tool NLL 16-18); v3 decoder not code-invariant -> leak. NEXT: tool factual both + dose_edit both on setA.

## Session 9 (runs 279-291)
- fwd_7: v4 BROKEN on tool (NLL 46): decoder term had constant offset ~145 inside LN'd q -> z-draw scale noise flips gates.
- v5 offset-free z-orthogonal decoders (fwd_8): lab/cuff fixed, art +0.016, finger broken (dmu ~+83).
- Probes P1/P2 (fwd_9/10, ratio method): tool decoded drug-1 value -29+-7.3 vs replica 0+-0.7 -> head decoding UNUSABLE; drug-2 gate does not open on tool.
- Run 291: head edits dropped; testing embed.numeric z_dose/z_rate column x3 (model's own dose pathway), judged by tool dose_edit + factual.

## Session 10 (runs 292-)
- Run 292: re-orientation dump (setA ids, fwd summaries). Current edit: embed.numeric dose/rate cols x3 (untested on tool).
- fwd_11 (dose_edit both setA, numeric cols x3): ins x2 -2.09 (orig -1.45), ins half +1.96 (+.82), ne x2 +.18 (+.15), ne half -.39 (-.41). Weak lever, NE unmoved.
- fwd_12 (factual both setA, x3): NLL 1.51901 vs 1.51903 -> no factual cost (numeric dose cols only feed drug START atoms).
- Run 293: numeric cols [2, 4] scaled x10.0 (from original). Next: dose_edit both + factual both on setA.
- fwd_13 (dose_edit both setA, x10): ins x2 -1.94, half +1.88, ne x2 +.20, half -.32 -> SATURATED (no gain over x3).
- Run 294: reverted to x3 on cols [2,4] (verified: factual neutral fwd_12, readouts fwd_11). FINAL candidate.
