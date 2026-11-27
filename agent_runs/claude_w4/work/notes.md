# Notes

## Run 1-2: read ARCHITECTURE.md, README. Dose enters only via encode_action (z_rate,z_dose,bins) -> u_k -> z_post & prelude.

## Runs 2-14: data facts
- All stays 25.17h; readings at integer hours, drugs at h+0.0833 (1h infusions). 30k stays; ins in 18.4k stays, NE in 13.1k.
- INSULIN POLICY (sliding scale on latest glucose reading; finger preferred if same hour as lab): rate=0 if G<=150,
  base 1 (150-180], 2 (180-220], 3 (220-260], 4 (>260), times iid lognormal noise (log-sd ~0.31-0.37) rounded to 0.5, min 0.5.
  Noise has ~zero autocorrelation (per-record). Finger checks become hourly when G>150.
- finger vs lab same hour: mean diff 0.03, sd 9.1 -> same latent glucose. arterial ~3 mmHg below cuff (sd 6.7).
- NE: rate 0.01..1.0 (cap at 1.0: 14% of NE records), dose=4.8*rate. NE on-prob falls smoothly with MAP; sharper near art MAP 64-65.
- INSULIN 1-HOUR EFFECT: within-band-noise OLS: -2.0 mg/dL per unit/h on next-hour finger glucose (se .05), stable across controls.
  RD at thresholds 150/180/220/260: -2.1..-2.4, -1.76, -2.0, -3.2..-4 (se .2-1.0) -> consistent, ~ -2 per unit/h at 1h.

## Runs 15-20: insulin dynamics
- glucose_lab measured randomly p=0.2/hour, independent of state -> unbiased outcome. finger hourly iff prev latest G>150 else p=0.3.
- 2SLS impulse response (dose surprises e=rate-E[rate|band] as instruments), LAB outcome, K=12:
  h=[-1.63,-1.83,-1.78,-1.4,-1.41,-0.89,-0.54,-0.71,-0.12,-0.33,-0.31,-0.03] (se~.3); cumsum ~ -11 at 12h.
  FINGER K=8: [-1.89,-2.42,-2.08,-1.66,-1.25,-0.9,-0.68,-0.48] (se .1).
- One-hour effect linear in rate (per-unit ~ -1.7..-2.3 for rates 0.5..5) and similar across glucose bands -> linear, homogeneous.

## Runs 21-29: NE policy & effects
- NE policy: start iff latest MAP (arterial preferred same hour) < 65 (h>0; hour-0 'starts' are arrivals on NE).
  On: log-ratio steps, drift depends on band (<65 up ~+0.05 additive, 65-75 hold, >75 down ~-0.04 & stop p~.19) AND smoothly on MAP value; iid noise (log-sd ~.3).
- MAP measured state-independently: art-line stays (14.5k, from hour 0) art p=.95, cuff p=.3; other stays cuff p=.9.
- Crossfit HGB expected-dose models (feats prev rate, latest reading, band, age, source, h) -> clean surprises e_ins2, e_ne2 (no within-band corr).
- INS->LAB K12: [-1.53,-1.79,-1.69,-1.32,-1.28,-.81,-.45,-.58,-.09,-.30,-.28,-.08] cum -10.2; INS->FINGER: [-1.83,-2.12,-1.90,-1.55,-1.08,-.76,-.44,-.28,-.21,-.08,-.12,.06] cum -10.3
- NE->MAP_ART (art stays) per mcg/kg/min: [3.71,2.71,1.34,1.06,.43,.79,.17,-.18,...] cum ~10; NE->CUFF non-art stays cum ~7; cuff in art stays cum ~10 (noisy).

## Runs 37-50 (after restart)
- fwd_1: ORIGINAL dose_edit (tuning sel100): ins x2 -0.348, ins x1/2 +0.208, NE x2 +0.018, NE x1/2 -0.245 -> model ~ignores doses.
- fwd_2: ORIGINAL factual sel100: value NLL 1.49383 (finger 1.5515, lab 1.2396, art 1.3522, cuff 1.6918). fwd_3: original states sel100.
- Vocab: padding rows (index 0) of bins/categorical are zero. predicted mean = sum p(b)*bin centre (16 coarse bins, top finger bin >258 centre 269).
  insulin rate z=(r-1.5)/1, bins edges .5,.5,.5,1,1,1,1,1.5,1.5,2,2.5,3,3.5,4,5 ; NE rate z=(r-.47)/.6.
- FACTUAL RESIDUAL TEST: regress (observed - old predicted mean) on insulin surprise s=r-E[r|band] lags1-8 (band_mean: 0,1.06,2.10,3.15,4.28):
  finger gamma=[-2.05,-1.64,-1.33,1.11,-.09,.34,.27,-.52] se~.6-.7 ; lab [-1.32,-.79,-1.22,...] se~1.2. -> old model ignores dose surprise; adding it improves factual fit.
- Tried to replicate model on CPU (runs 44-49): could not match z posterior (mean err ~.04 vs cross-stay sd .07). ABANDONED replica.
- States (fwd_3) ridge CV R2: out->latest glucose .95-.98, ins rate lag1 .76 lag2 .71, NE rate .46-.57, but insulin SURPRISE s1,s2 R2~0.
  -> state keeps only band-predictable dose info; heads-only edit impossible; must inject surprise into core.

## Runs 57-64: PROBE (fwd_4) + exact encode_action replica
- Probe: z_post rewritten so z_mean copies input coords (offset 30); z_in.weight=0. Activation = GELU (act(.5)=.3457).
- z_post input order [ctx, e_k, u_k] confirmed. u_k for no-action bundles = const = LN(mlp2(size.bias)) (not zero).
- u_k depends ONLY on drug rates in bundle (END atoms not in action set). Insulin atom EXACT (err 0):
  code[8]+kind[0]+ekind[2]+unit[3]+rate_unit[2]+kq[0]+bins.rate[b]+bins.dose[b]+numeric([0,0,z_d,1,z_r,1])+bias;
  bin b = 1+#(edges<x); z=(x-median)/iqr clip 10. DeepSets: sum mlp1/sqrt(n) + size(log(1+n)), mlp2, LN eps1e-5.
- u_I(rate) is non-monotonic: rate 1 very different from .5,2,3,4,6 (which are similar) -> weak dose signal.

- ctx = out_prev + out_norm(mean_slots(h_prev*exp(-dt_s*exp(flow.a)))) + gap(log(dt_s)/10)  [EXACT, dt in seconds]
- e_k at drug bundles EXACT: encode({END atoms}); END atom = code[c]+kind[2]+ekind[2]+kq[0]+numeric.bias; END for records started 1h earlier.
  All drug records start at h+0.0833. NE atom: unit[4], rate_unit[3], dose=4.8*rate. Measurement-atom e_k NOT replicated (err ~.33) - not needed.
- z posterior encodes reading surprise: R2(z->surprise) finger .85, lab .73, art .79, cuff .77 (glucose dims 20,27,9,30; MAP dims 13,22,2,23).
- PLAN: repurpose z_post hidden units gated on u_k (drug bundles only) to add kappa*d_news*(rate - E[rate|ctx]) to z mean.

## Runs 79-84 (restart 2)
- dose_edit summary mean_shift = POOLED mean over 'reached' rows (stay-hours with an outcome reading, after drug ran from 12).
- IRF-based TRUTH on sel100 reached rows (hF finger, hA art): insulin x2 -12.96, x1/2 +6.48 (x0 +12.96, x1.5 -6.48, x3 -25.9);
  NE x2 +4.94, x1/2 -2.47 (x0 -4.94, x3 +9.88). Original model: -0.35, +0.21, +0.02, -0.25.

## Runs 85-100: INJECTION EDIT v1 (z_post only)
- u_k (encode_action output) loses insulin magnitude via LayerNorm (sigma(y) 17..119), but a ridge readout from u IS exact for
  insulin-only (rms .0002, rates<=11.5; z clip at 10 => rates>11.5 indistinguishable). Joint readouts fI (I-only+combos), fN:
  mean error of readout change over tuning drug bundles ~1 pct (ins, all m), NE <1 pct (m=.5,2), 5 pct m=0, -10 pct m=3.
- Gates: LinearSVC on u: insulin present (on>=.886, off<=-.355), NE present C=1000 (on>=.736, off<=-.001). thresholds = midpoints.
- E[rate|ctx]: ridge on CTX3 at drug-present bundles (fwd_3), CV R2 ins .56, NE .31 (model's own knowledge).
- z direction: regress z post mean on reading surprises: finger==lab direction (cos 1.0), art==cuff (cos 1.0); per-unit norms
  finger .0233/mgdl, lab .0094, art .069/mmHg, cuff .045. Use D4[:,0] (finger) for glucose, D4[:,2] (art) for MAP.
- z_post units repurposed (low worst-case UB, dead at drug bundles): [110, 183, 83, 149] = [A_ins, gate_ins, B_ne, gate_ne].
  unit pairs GELU(P+x)-GELU(P), P=M*(gate-th), x=c*(f(u)-E(ctx)); dz = -kapI*s_ins*D4[:,0] + kapN*s_ne*D4[:,2].
- v1 saved via build_edit(2.0, 4.0, units4) (cI=.01,cN=.05,MI=40,MN=60). Replica check at drug bundles exact (1e-5).

## Runs 104-122 (restart 3): v1 FAILED; prelude attention mapped
- fwd_5 = v1 (z_post injection) dose_edit tuning sel100: ins x2 -0.349, x1/2 +0.209, NE x2 +0.018 -> identical to original. v1 useless.
- WHY: fwd_4 probe (z_in.weight=0) vs original states: out rel change 0.2%, h <1% -> z token barely used by state update.
- Prelude CPU replica (pre-norm, 4 heads x48, kv rows 0:192=k,192:384=v, exact GELU, LN eps1e-5). Slot->token attention:
  drug bundles: head0 slots .95/e .05; heads1,2,3 -> u_k token .998/.997/.992; z token 0 everywhere.
  measurement bundles (approx e_k): head0 slots .91/e .09; HEAD1 -> u token 1.00 (min 1.0); head2 slots .99; head3 slots .6/u .28/e .11.
  => HEAD 1 = clean always-on u_k reader: output = Wo[:,48:96] V1 LN1(u_k), state-independent. No-action u_const = LN(mlp2(size.bias)) (norm 11.6).
- PLAN v2: carrier via head-1 value path (rank-1 add to kv rows 240:288) and/or prelude FFN units (gate+state E) to inject kappa*(rate - E[rate|state]).
  Factual fit needs subtraction of E[rate|state]; linear readout weak (R2 .56) vs band ceiling ~.8 -> consider step units on latest-G readout.

## Runs 123-143 (restart 4)
- Runs 123-139 built a CPU replica of a full core step at drug bundles (functions mha, ffn, age_emb, mha_m, build_mem, step_full,
  step_var, step_parts, coda_only, fitstat, step_cv; targets yt (2319,4,192) new slots, ot (2319,192) out from fwd_3 states).
  Variant comparison (run 139): memory cross-attn with age embedding added to keys AND values ('ageKV') fits best:
  err 39.0, R2 (y .998, out .994); lnM 210.8 (.879,.873); noage 252.5 (.836,.827); lnM_ageKV 78.6 (.998,.979).
- Source of replica functions not recoverable (no In history). Signatures: step_full(rws,E,htf,incl=True,unit=3600.,pre=False,dE=None).

## Runs 156-160 (restart 5)
- Model persistence (fwd_2 factual, pred_{t+1} ~ obs_t + pred_t): finger c_obs .554 c_pred .403 (sum .958); art .644/.323 (.966); cuff .39/.57.
  => a belief-level shift persists ~0.96/h in rollout (true insulin IRF decays by lag ~9). Injection gain must be calibrated by forward
  dose_edit against truth (-12.96/+6.48 ins, +4.94/-2.47 NE on sel100); shape error expected for start 6/16.
- nuG/nuM (4,192): per-mg/dL (per-mmHg) direction in new slots y from reading surprise (same in all slots, norm ~1.02 / 5.3 per slot).
- PLAN v2: prelude FFN unit pairs at drug bundles add delta*(gate)*(f(rate)-E[rate|state]); choose delta so dy ~ -g*s*nuG/0.554.

## Restart 6 (run 190-192)
- Runs 161-189 were not noted. Kernel has build_v3(gI,gN,sP): head-1 value-path carrier (kv rows 240:288, out cols 48:96; dirs aI3,aN3,aP3; readouts qd['tI','fN','pI']; inject dirs dI,dN). Never forward-tested.
- Run 189: readouts tI/fN vs dose multipliers: bias <.05, rms ins .06-.2, NE .03-.11; meas bundles ~0.
- Run 192: dropped v1 z_post edits (useless).

## Restart 7 (runs 198-204)
- build_v3(gI,gN,sP) decoded: kv rows 240:288 += outer(aI3,BB3.tI)+outer(aN3,BB3.fN)+outer(aP3,BB3.pI); Wo[:,48:96] remapped so
  aI3 -> -gI*dI, aN3 -> +gN*dN, aP3 -> +sP*qd (a-dirs orthonormal; original value proj on them ~0, sd .001 at drug bundles).
- Readouts on LN1(u) at drug bundles: tI = insulin rate (rms .11); fN = (NE rate - 0.52)*present (rms .05), ~0.03 when absent;
  BB['tN'] = uncentred NE rate (rms .05); pI = insulin presence (sd .005). Slot norm ~795, |dI| 13.2, |dN| 28.7.
- Stored Jm/nuG arrays: semantics unclear (Jm@dI only cos .17 with -nuG) -> rely on forward passes for calibration.
- fwd_6 plan: dose_edit both, sel100 tuning, v3(gI=1,gN=1,sP=0). Truth sel100: ins x2 -12.96, x1/2 +6.48; NE x2 +4.94, x1/2 -2.47.

## Restart 8 (runs 252-)
- Runs 205-251 were NOT noted (lost); run 251 crashed (memory). Current edit = v3(gI=1,gN=1,sP=0) == /work/v3_g1_1_0.npz (kv rows 240:288, wo cols 48:96).
- fwd_6 (dose_edit both, tuning sel100, v3(1,1,0)): EDITED ins x2 -0.145, x1/2 +0.066, NE x2 +0.010, NE x1/2 -0.163
  vs ORIGINAL -0.348/+0.208/+0.018/-0.245. => v3 at gain 1 slightly WEAKENS dose response (truth -12.96/+6.48/+4.94/-2.47).
  stay 2020000064 h13 ins x2 shift edited +0.99 vs orig -0.10 (sign wrong?).
- fwd_7: states both (edited=v3(1,1,0)) tuning sel100, 5218 bundles: use to see how injection propagates.

- Run 255: v3 effect on state is tiny: median |d out| .004 (out norm 13.9), |d mean_h| .2-.3 at drug bundles (h slot norm 560-1100).
- Run 256: softplus(delta_in)~1 (.90-1.07), a_bar .30-.46 (h keeps ~37%/step); core.slots norm <1 (slot tokens ~= decayed h).
  b_in, c_out sv 5.5..0.001 (top ~100 dirs usable). prelude_norm w .8-1.1, b ~0. Value head reads out=LN(mean y) and LN(mean h flowed).
- Run 257 (an1.npz): LN slot-token sigma 27-57 (median 40) -> channel via prelude_norm has 1/sigma gain variation.
  cov(LN slot tokens): 124 eig<1e-4, 152<1e-3 (state lives in ~40 dims) -> many clean channel directions.
  News directions (dY per unit surprise, same vector in all 4 slots, cos 1.0): finger .82/mgdl, lab .92, art 5.6/mmHg, cuff 3.0;
  cos finger-lab .51, art-cuff .905, finger-art -.18. Saved nu (event,slot,dim) order [finger,lab,art,cuff].
- PLAN v4 'dedicated channel': head-1 out-proj maps insulin readout -> c*q (small c, q in slot null space, zero-mean, low FFN sensitivity);
  core.b_in += t (x) w reads q from E and writes t = C_out^-1(-k*nuG) (b_in has no bias; w.E_nat must be ~0). Same for NE with nuM(art).

## v4 'dedicated channel' BUILT (runs 259-261) -- files /work/rep_drug.npz, /work/chan.npz, /work/v4_parts.npz
- Replica (run 259) exact at drug bundles: attention head0 slots .946/e .054, heads1-3 on u .998/.997/.992.
- v3 kv diff (rows 240:288) = rank 3, A = top-3 left sing vecs; zA = A^T Dkv LN1(u) = M f, f=[rI, (rN-.52)*present, insulin presence]
  (fit rms .11/.025/.005); orig head-1 value along A ~0 (sd .001), ~0 at u_const. Keep kv = v3 kv; only remap out-proj.
- Channel dirs qs (chan.npz) = low-variance dirs of natural LN(prelude out x'') (var ~5e-7), orthogonal to 1, mean, beta/gamma; low FFN sensitivity.
  Injection c=6 along q gives dE = c f/sigma'' along gamma*q exactly (offtarget <1%); sigma'' median 80.5; natural w.E sd .0008.
- b_in edit: Wb = Wbin + sum_j outer(k_j t_j - Wbin qE_j, w_j), qE=gamma_p*q, w=q/gamma_p (replaces natural response along channel).
  t = (sig_ref/c) * C_out^-1 J_coda^-1 (target)/softplus(delta_in); coda inverted on states (exact), Jacobian Newton-solved.
- Targets: insulin tgt_I = -0.87*unit(unit(nuF)+unit(nuL)) (~ -1 mg/dL news per unit (rI - rbar)), rbar=1.77 (presence-centred);
  NE tgt_N = +4.3*unit(unit(nuA)+unit(nuC)) (~ +1 mmHg news per unit (rN-.52)). Replica: dy per unit .90 (cos .9986) / 4.54 (cos .99997).
- build_v4(kI,kN): head-1 out TA = c*[qI, qN, -rbar*qI] M^-1; Wo[:,48:96] = Wo(I-AA^T)+TA A^T. v4(1,1) SAVED (b_in, kv=v3 kv, out).
- NEXT: fwd dose_edit both sel100 for v4(1,1) -> rollout gain per k; then factual both for NLL.

## Restart 9 (runs 266-)
- fwd_8 = v4(1,1) dose_edit both on 100 tuning stays (NOT sel100; 30 overlap). EDITED ins x2 +3.30, x1/2 -5.08, NE x2 +0.23, x1/2 -0.18; ORIGINAL -0.38/+0.21/+0.10/-0.31. Per-hour ins x2 shifts all positive (2-5), x1/2 negative growing to -9.4.
  Factual (recorded-dose) rollout drift edited-orig grows +4 (h13) -> +26.5 mg/dL (h24), positive almost everywhere; NE drift +0.3->+2.7.
  => v4 insulin effect has WRONG SIGN and/or a dose-independent leak. Checking with record regression (run 269).
- Run 269: drift NOT from insulin injection: rows with no insulin after h12 drift +25.2; drift ~ 3.1 + 1.96*dh - 0.60*cen. x2 shift ~ 7.6 - 0.42*sumr (dose part right sign, weak).
  TRUTH on fwd_8 stays (IRF): ins x2 -13.17 (by hour -3.4..-18), NE x2 +4.23.
- Run 270-271: head-1 path clean at u_const (f=[.0016,-.0009,.0003]). prelude_b(T,Wkv,Wo)->(x1 after attn, x2 after FFN) EXACT vs rep_drug.
  Channel coord s=q.LN(x2): drug bundles mean 0 sd .0006; MEAS bundles (Tm, approx e_k) chI mean .0097 sd .0044, chN -.001 sd .005 -> leak at meas bundles.
  WARNING: global c was overwritten in run 270 (restored from v4_parts in 272); global rbar is 0.52 (v4_parts rbar checked in 272).
  build_v4(kI,kN,cent=True,save=True) uses globals qs,c,rbar,M,Wo,A,Wbin,tI,tN,gp,kv3. st8 = fwd_8 stays saved /work/st8.json.

## Restart 10 (run 277-)
- Runs 272-276 outputs were blocked by a content filter: print only aggregates, never raw rows or stay ids.
- fwd_9 = factual both tuning 100 stays for v4(1,1): value NLL 1.4938 -> 1.610 (+0.116!). finger 1.5515->1.6887.
- Cause (run 278): v4 head-1 readouts B fit only on u tokens; on slot tokens f~300+-760, reading tok ~+-3000 -> 0.2% attention leak = +-1 rate unit noise everywhere.
  Tm token 4 == u_const (f~0); b_in natural channel activity tiny (sd .0008).
- Run 279: v6 = ridge refit of B (targets on drug u; ~0 on slots/e/tok5/u_const), build6(B,kI,kN,rbI) re-implements v4 (repro checked). wo=None. Saved v6(1,1) if sel not None.
- Run 280 grid: lam=1e-4,wo=1e-4 -> drug err [.103 .052 .012], insComb .097, slot readout rms 1.2, e/t5 1.5 (v4: 760/3000).
- Run 281: SAVED v6(1,1) = build6(B,1,1), B in /work/v6_parts.npz (prelude_b now crashes: some global clobbered; not needed).
- fwd_10 = dose_edit both sel100, v6(1,1): EDITED ins x2 -1.422, x1/2 +.696, NE x2 +.102, x1/2 -.374; ORIG -.348/+.208/+.018/-.245.
  => leak fixed, sign right & ~linear, but chain gain only ~.18 (ins) / ~.03 (NE) of design. Need kI~12, kN~58 with v4 targets.
- Run 282: v7 = build7(B, tI5, tN5, kI, kN): v5 targets (predicted-mean readouts wF..wC, sig_m 58.6). SAVED v7(1,1). Next: dose_edit sel100.
- fwd_11 = dose_edit both sel100, v7(1,1): EDITED ins x2 -2.948, x1/2 +1.411, NE x2 +.584, x1/2 -.742 (orig -.348/+.208/+.018/-.245).
  => added per k: ins -2.60/+1.20 (linear), NE +.566/-.497 (log-like asym). kI~4.85 fits ins; NE compromise kN~6.9 (x2 3.9, x1/2 -3.65).
  Rough factual-cost estimate at full gains ~.005 (ins) + ~.0045 (NE) -> may need to scale down. Constant centering needed for sustained effect.
- Run 283: SAVED v7(4.85,6.9). Next: factual both sel100 + dose_edit both sel100.
- fwd_12 = factual both sel100, v7(4.85,6.9): NLL 1.4938 -> 1.7479 (+.254). finger +.166 lab +.393 art +.374 cuff +.175.
  dp rms finger 15.3 (res sd 15.0), lab 21.7 (12.8), art 6.6 (5.0), cuff 5.7 (7.7); corr(dp,res0) +.135/.148/.203/.116 => first-order term helps.
  Rollout persistence of injected pulses is weak (~20%/h of immediate). Cost model per event dNLL(s)=a s+b s^2 (s=k/k_full) -> zero near s~.25-.3.
- Run 285: SAVED v7(1.455,2.07) (s=.3). Next: factual + dose_edit both sel100.
- fwd_13 = factual both sel100 v7(1.455,2.07): NLL +.173 (finger +.038 lab +.133 art +.301 cuff +.225). dp rms 9.5/16.2/6.4/6.7 (s=1: 15.3/21.7/6.6/5.7)
  => large GAIN-INDEPENDENT perturbation (finger ~8.7, lab ~15.5, art ~6.4 rms); dn median ~.001 (heavy tail). Gain-dep part: finger ~12.6*s, lab ~15.2*s.
  Hypothesis: channel injection magnitude c=6 into prelude x1/E (fixed) disturbs state. Fix: c->0.4, b_in gain x15 (same b_in response).
- Run 287: build8(B,tI5,tN5,kI,kN,cc). SAVED v9 = build8(...,1.0,2.0,0.4). Next: factual + dose_edit sel100.

- run289 was a syntax error (lost). run290: fwd_14 = factual v9: NLL +0.234. fwd_15 states v9: perturbation tiny at drug/lag1, grows with lag, HUGE before first drug (|dh|~|h|).
- fwd10 insulin_half add/h: +0.1 +0.2 +0.2 +0.3 +0.4 +0.4 +0.5 +0.6 +0.6 +0.8 +0.8 +0.7
- fwd10 insulin_x2 add/h: -0.4 -0.5 -0.7 -0.8 -0.9 -1.0 -1.2 -1.2 -1.3 -1.4 -1.5 -1.7
- fwd10 ne_half add/h: -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.1 -0.2 -0.2 -0.2 -0.2
- fwd10 ne_x2 add/h: -0.0 -0.0 +0.0 +0.0 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.1 +0.2
- fwd11 insulin_half add/h: +0.4 +0.5 +0.6 +0.8 +1.0 +1.1 +1.2 +1.5 +1.6 +1.7 +1.8 +1.7
- fwd11 insulin_x2 add/h: -0.8 -1.3 -1.5 -1.9 -2.1 -2.3 -2.7 -2.8 -3.1 -3.6 -3.7 -4.4
- fwd11 ne_half add/h: -0.2 -0.3 -0.4 -0.5 -0.5 -0.5 -0.5 -0.5 -0.6 -0.7 -0.7 -0.7
- fwd11 ne_x2 add/h: +0.0 +0.1 +0.3 +0.3 +0.5 +0.6 +0.6 +0.8 +0.8 +0.8 +1.0 +0.9
- stays 100, with drug 100; n_atoms mean all 2.53
- fwd12: cover 1.00 dNLL +0.2540 = pre +0.2451 (n547) + post +0.0090
-    glucose_finger: dNLL pre +1.889 post +0.008 | dp_rms pre 52.0 post 2.5
-    glucose_lab: dNLL pre +3.581 post +0.002 | dp_rms pre 65.5 post 1.7
-    map_arterial: dNLL pre +3.878 post +0.009 | dp_rms pre 21.2 post 1.4
-    map_cuff: dNLL pre +1.228 post +0.018 | dp_rms pre 14.9 post 1.9
- fwd13: cover 1.00 dNLL +0.1728 = pre +0.1746 (n547) + post -0.0018
-    glucose_finger: dNLL pre +0.479 post -0.002 | dp_rms pre 32.5 post 0.8
-    glucose_lab: dNLL pre +1.227 post -0.001 | dp_rms pre 49.0 post 0.6
-    map_arterial: dNLL pre +3.239 post -0.004 | dp_rms pre 20.9 post 0.4
-    map_cuff: dNLL pre +1.721 post +0.001 | dp_rms pre 18.6 post 0.6
- fwd14: cover 1.00 dNLL +0.2343 = pre +0.1727 (n547) + post +0.0616
-    glucose_finger: dNLL pre +1.698 post +0.001 | dp_rms pre 51.6 post 1.0
-    glucose_lab: dNLL pre +4.510 post +0.005 | dp_rms pre 72.6 post 1.8
-    map_arterial: dNLL pre +1.403 post +0.137 | dp_rms pre 16.3 post 2.5
-    map_cuff: dNLL pre +1.055 post +0.109 | dp_rms pre 13.5 post 3.0

- run290 KEY: factual cost is ~all from readings BEFORE first drug bundle (n547): v7 s=1 pre +.245/post +.009; v7 s=.3 pre +.175/post -.0018; v9 pre +.173/post +.062.
  => v7 post-drug cost model c(s)~ -.0124 s + .0214 s^2 (zero s~.58, s=.7 -> +.002). Need to kill the PRE-DRUG LEAK. v7(1,1) ins x2 added/h -0.8..-4.4 (accumulates like truth).
- run291: diagnosing head-1 readout on slot/z/u_const tokens pre vs post.

- run291/292: global P (weight prefix used by LN1/prelude_b) was clobbered by my print helper; restored P=None. Funcs: prelude(Xs4,e,u,z,Wkv,Wo), prelude_b(T,Wkv,Wo), chs(T,Wkv,Wout), enc(atoms,pre), build7(B,tIv,tNv,kI,kN,rbI). No source available.
  PLAN: refit B on attention-weighted token mixtures at ALL bundles (targets 0 where no drug) to kill pre-drug leak; then build7 at s~.6.

- run293: LN1 replica exact (eps1e-5), head-1 attention replica exact (1/sqrt48). Td=[4 slots,e,u,z]; Tm=[4 slots,u_const,tok5]; slot tokens != h+slots (decayed h). Globals Wkv/Wo/Wbin = original.
- run294: refit B on attention-weighted mixtures (drug targets F; zero at meas bundles x3, synthetic first bundles x20); cands saved /work/v10_cands.npz
  map cos Tm 0.797 Td 0.990 | F0~rI corr 1.000
  Tm5 mean dist to Td-z 0.98, to Td-e 12.98 -> Tm5 is z
  curB: drugErr [0.106 0.056 0.015] measPre [0. 0. 0.] measPost [0. 0. 0.] first [126.35  101.536  19.814]
  h1 att meas pre: slots 0.000 uconst 1.000 | post: slots 0.000 uconst 1.000 | first: slots 0.243
  lam1e-07: drugErr [0.151 0.081 0.023] measPre [0.001 0.001 0.   ] measPost [0.001 0.    0.   ] first [0. 0. 0.]
  lam1e-06: drugErr [0.218 0.136 0.052] measPre [0.001 0.    0.   ] measPost [0. 0. 0.] first [0. 0. 0.]
  lam1e-05: drugErr [0.308 0.201 0.091] measPre [0. 0. 0.] measPost [0. 0. 0.] first [0.002 0.001 0.001]
  lam0.0001: drugErr [0.419 0.238 0.144] measPre [0. 0. 0.] measPost [0. 0. 0.] first [0.    0.    0.004]

- run294 KEY: leak origin = FIRST bundle of each stay: slot tokens=core.slots (h=0), head1 attends 24% to slots, B readout there = [126,102,20] (!). All other non-drug bundles: head1 attends 100% to u_const, readout 0.
- run295: v10 = B projected orthogonal to LN(core.slots) (+e-token mean/PCs), build7(Bsel,tI5,tN5,2.91,3.45) (s_I=.6,s_N=.5). B in /work/v10_parts.npz
  first-bundle(with e proxy) h1 att: slots 0.242 e 0.190 uconst 0.000 z 0.568
  curB: drugErr [0.106 0.056 0.015] first [126.35  101.536  19.814] first+e [126.213 101.503  19.799] e tok [0.292 0.251 0.76 ]
  proj ke0: drugErr [101.571  53.915  14.814] meas [146.091  37.06   20.877] first [52.076 48.823  2.33 ] first+e [42.83  39.009  2.218] e tok [26.127  5.014  4.189]
  proj ke5: drugErr [139.733  29.29   19.565] meas [149.541   9.568  22.805] first [102.186  11.433   8.574] first+e [81.445  9.112  6.833] e tok [0. 0. 0.]
  proj ke10: drugErr [118.462  46.563  33.997] meas [108.704   7.991  11.527] first [82.127 14.615  7.715] first+e [65.457 11.649  6.149] e tok [0. 0. 0.]
  proj ke20: drugErr [107.714  54.366  17.045] meas [143.312  17.744  32.942] first [105.372  59.767   7.44 ] first+e [83.984 47.636  5.93 ] e tok [0. 0. 0.]
  proj ke40: drugErr [150.613 114.699  36.598] meas [88.686 84.132 31.272] first [ 3.903 83.033  2.479] first+e [ 3.111 66.179  1.976] e tok [0. 0. 0.]
  chosen k_e 0
  build7 ret tuple 3 | kv hash 80745ab7 -> 80745ab7 | edited ['core.b_in.weight', 'core.prelude.0.attn.kv.weight', 'core.prelude.0.attn.out.weight']
  kv struct err 0.12
  |dWb| 5.99e+03 |dWo| 1.32

- run295: projecting B fails (B large-norm w/ cancellations -> errors ~100). build7 RETURNS (kv,out,b_in) tuple, does NOT save.
- run296: v10 = min-change fix of B (zero readout on first-bundle mixtures, preserve drug readouts in drug-data metric) + build7(Bn,tI5,tN5,2.91,3.45) saved manually. Bn in /work/v10_parts.npz.
  fixB: drugErr [0.107 0.055 0.015] meas [4.104 0.222 0.236] first [0. 0. 0.] first+e [0.054 1.476 0.238] e tok [0.175 4.811 0.777] |B| 388->386
  tuple [(384, 192), (192, 192), (192, 192)] n_kv 1 n_out 1 n_bin 1
  SAVED v10. kv struct err 9.1e-09 | off-A kv change 1.4e-06 | rows outside 240:288 changed 0 | |dWb| 696 |dWo| 13.3 | hash cabbb37c ['core.b_in.weight', 'core.prelude.0.attn.kv.weight', 'core.prelude.0.attn.out.weight']

- run296 v10 was BAD: min-change fix ignored u_const -> insulin readout 4.1 at every meas bundle. run297: v10b = joint WLS fix (keep drug readouts, zero meas/u_const/first(+e proxy), keep e-token readouts) + build7(...,2.91,3.45) saved manually if checks pass.
  fixB2: drugErr [0.107 0.056 0.015] meas [0. 0. 0.] uconst [[ 0. -0. -0.]] first [0.    0.001 0.001] first+e [0.001 0.002 0.002] e tok [0.011 0.037 0.009] |B| 386
  good True
  SAVED v10b. kv struct err 9.9e-09 | |dWb| 696 |dWo| 13.3 | hash cf96d21b ['core.b_in.weight', 'core.prelude.0.attn.kv.weight', 'core.prelude.0.attn.out.weight']

- RESTART@345: run298 = info dump (ids, truth lines, vars) + v10b backup /work/v10b_tensors.npz. PLAN: fwd factual both tuning (4x100) + dose_edit both sel100 on v10b. If dNLL>+.003: run299 rebuild weaker via build7; run300 reserve (fallback drop_edit).

- fwd16 = factual both sel100 v10b: NLL 1.4938 -> 2.0104 (+.517; finger +.56 lab +.82 art +.48 cuff +.38). v10b FAILS; carrier design fragile off drug bundles.
- run299: DROPPED all edits (back to original weights) to guarantee factual fit; v10b kept in /work/v10b_tensors.npz. Next: forward factual both to confirm, then FINAL.
