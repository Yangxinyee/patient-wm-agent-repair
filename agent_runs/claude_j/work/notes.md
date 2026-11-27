
# NOTES (world-model dose repair)

## Data facts (train.csv, 30k stays, hours -2..25.17; readings at integer hours, infusions at h+0.083 lasting 1h)
- Insulin: sliding scale on glucose_finger (GF) read 5 min before: GF<=150 none; (150,180]:1, (180,220]:2, (220,260]:3,
  (260,300]:4, >300:6 units/h, times iid lognormal multiplier (sd log ~0.31, mean rate per base 1.048/2.09/3.15/4.21/6.29),
  rounded to 0.5. Multiplier: no autocorrelation, uncorrelated with covariates -> exogenous instrument. dose==rate.
- GF measured every hour if last GF>150, else ~30% random. glucose_lab (GL) random 20% of hours. GF and GL same latent
  glucose (GF-GL mean 0, sd 9).
- NE: start iff map_arterial(MA)<65 (or MC if no MA) at h>=1 (hour-0 starts are partly random). Once on: titrated
  multiplicative random walk: log ratio step ~ +0.1 (MAP<65), ~-0.02 (65-75), ~-0.2..-0.3 (>75) + iid noise sd~0.29,
  rounded 0.01, capped 1.0; stops only when rate <~0.05. NE dose=4.8*rate. ~48% stays have arterial line (MA ~91%
  of hours, MAP-independent); others map_cuff (MC) ~86% of hours. MC-MA = +3 mmHg, sd 6.7.

## Causal effects (estimated from exogenous dose noise; RD at thresholds agrees for insulin)
- Insulin fixed-dose impulse response on glucose (per 1 unit/h for one infusion-hour started at h+0.083), at hour h+k:
  GF-based: [-2.00,-2.38,-2.08,-1.70,-1.16,-0.91,-0.65,-0.49,-0.25,-0.11,-0.09,0.01]; GL-based similar
  [-2.08,-2.45,-2.24,-1.52,-1.44,-1.02,-0.65,-0.43,...]. ~= -2*k*0.6^(k-1) (sum -12.5). Linear in dose, same across bands.
  RD at 150/180/220: one-hour effect -2.0/unit.
- NE impulse response on MAP (per 1 mcg/kg/min for one hour), MA-based: [3.3,2.55,1.69,0.92,0.37,0.21,-0.1,0,...]
  cum +8.8; MC-based similar (cum ~9.5). ~= 3.3*k*0.39^(k-1).

## Architecture summary (from /model/ARCHITECTURE.md)
- atoms -> embed (code+kind+unit+bins.value/dose/rate (17 rows shared across codes, per-code edges) + numeric@[zv,hv,zd,hd,zr,hr]+bias)
- measurements -> encode.* (DeepSets) -> e_k ; insulin/NE atoms -> encode_action.* -> u_k (0 if none)
- state: h (4 slots x192), memory M (128 last e's), out(192). Step: time-flow decay h*=exp(-dt*exp(core.flow.a));
  z posterior from [ctx,e_k,u_k]; prelude transformer over [h+slots, e_k, u_k, z_in(z)] -> E (prelude_norm);
  blocks(h+E) -> u ; h <- a_bar*h + softplus(delta_in)*b_in(E) + (u-(h+E)); coda: y=coda(c_out(h)); h=y; out=out_norm(mean_slots(y))
- heads: view = out + out_norm(mean_slots(h after flow)) + gap(log dt) + z_head(z); q=query_norm(view+type_emb[type]);
  with_code = q + embed.code[code] + last_numeric([has_last, log(1+age)/10, z_last_value, z_last_rate]) + bins.value[last]+bins.rate[last]+last_status
  value logits = attr_heads.value_bin(with_code) + embed_transition.value_bin[code*17+last bin]; mean = sum p*centre.
- dose_edit: record to hour 12 as is; from 13 readings hidden & sampled; insulin atoms from hour 12 on get rate&dose x2 (z and bins change).
## Forward pass 1 (original, dose_edit, tuning, 100 stays in sel1.json)
- model: ins x2 -2.75, ins x1/2 +1.38, NE x2 +0.69, NE x1/2 -0.58 (n stay-hours 818 / 693)
- truth table R (kernel) uses KI=-2k*0.6^(k-1), KN=3.3k*0.39^(k-1) (per unit/h resp. per mcg/kg/min, per infusion hour)

## Session 2 findings (runs 29-42, fwd 2)
- fwd_2: factual original, tuning sel1 (100 stays): value NLL 1.51162 (n=5214); GF 1.5057, GL 1.31, MA 1.3501, MC 1.7898.
- TRUTH on sel1 (kernel KI, KN): ins x2 -13.95, x1/2 +6.98, NE x2 +4.28, x1/2 -2.14. Model (orig): -2.75, +1.38, +0.69, -0.58.
  Model/true ratio ~0.26 at lag1 and ~0.16-0.23 later (insulin); NE lag1 ratio 0.11.
- Lag-1 effect re-checked on train, ONLY hours where insulin given (base>0): slope -1.93 (se .05) per unit/h
  (bands -1.86/-2.01/-1.95/-2.36). [Including GF>150-but-no-insulin rows biases to -1.0: ~8.8k such rows.]
- Model's factual within-band dose-deviation slope at lag1 (given-only, tuning 100 stays): -0.57 (se .31) vs truth -1.93.
  => model under-uses dose ~27%; its reading dynamics absorb expected insulin. Picture: pred_old ~ N(g) - 1.46*c(g) - 0.54*d,
  truth N(g) - 2d. Needed edit term: -1.46*sum_j a^(t-j)*(dose_j - c(GF_j)) (zero-mean on factual data; c=E[dose|GF]).
- vocab: z=(x-median)/iqr clipped +-10 (LINEAR). insulin dose/rate median 1.5 iqr 1; NE rate median .46 iqr .59, dose 2.208/2.832.
  fine bin = 1 + #edges below x. GF edges [78,87,96,109,128,143,154,164,173,183,195,214,230,243,260] (not aligned w/ 150/180/220).
  GF centres [73..272] -> predicted mean bounded 73..272. NE rate edges cap at 1.0 -> x2 NE rates (>1) unseen bins.
- Plan: build CPU replica of the model (validate vs 'states' request), then engineer a compensated dose channel.

## Session 4 (runs 90+): state of kernel
- Kernel holds an earlier replica of the per-bundle state update (fns: build_bundles, bundle_vecs/bvd, encode, step, mha,
  tlayer, block, ffn, one_step_err; dicts W/Wt of weights; BUN = bundles of the 100 sel1 tuning stays; Xt = fwd_3 states).
  Source is NOT recoverable (defined in kernel). o_chk=8.9e-7 suggests out matched. No head implementation yet.
- fwd_3 = states, tuning sel1 100 stays, 5116 bundles (x: out | 4 h slots | z post mean).
- KEEP OUTPUTS SHORT (content filter refused long raw listings). Save new code to /work/*.py files.

- HEAD REPLICA (/work/heads.py, runs 99-103): value logits = value_bin.2(gelu(value_bin.0(with_code))) + embed_transition[code*17+lastbin];
  q = query_norm(view + type_emb[1]); view = out + out_norm(mean_slots(h*exp(-dt*exp(flow.a)))) + gap(log(dt_s)/10) + z_head(z), z~prior;
  last_numeric cols: 0 = has_last, 3 = log1p(age_s)/10, 6 = z_last_value (others 0 for measurements); last_status NOT added; bins[0] rows zero.
  MAE vs tool predicted mean: 0.94 (readings with a last value), 2.3 on first readings (only first GF after admission is off).
- Dead hidden units in value_bin.0: 89, 170, 149 (max contribution <0.015 logits). PLAN: repurpose 89 (insulin->GF/GL) and 170 (NE->MA/MC),
  gated by embed.code, linear regime, constant offset cancelled in embed_transition rows; readout fit by ridge on q.

## Session 5 (runs 112-126)
- No source history; replica fns are black boxes. BUN keys int; maps from hourly_maps(tus) are nested dict[stay][hour].
- fwd factual csv row order != reading_table order: MERGE on (stay_id,hour,event).
- Drug bundles are separate (h+0.083), readings at h. Heads average probs over 4 prior-z draws.
- Linear readout from head query q (100 tuning stays): CV R2 RI .89, exogenous part TI .15, RN .62.
- Right after drug-A bundle, dose-deviation CV R2: out .07, hmean .38, z .18 -> deviation weakly encoded.
- PLAN: replica counterfactual states (doses scaled, readings recorded) -> joint ridge readout (factual+cf) -> head unit edit.

## Session 6 (run 147+): restarted; runs 127-146 undocumented (last: probe end->u attrs). Re-orienting.
- Run 161-163: replica validated: step(out_p,h_p,dt_s,e,u,z,Mem,ages): Mem = e of ALL bundles 0..k (incl current, last 128), ages seconds; one-step out err med .01 (GPU precision).
  zpost: ctx=out_p+out_norm(mean_slots(flow(h_p,dt)))+gap(log(dt_s)/10); z=z_post.2(gelu(z_post.0([ctx,e,u])))[:32] -> err ~0 (dt seconds). vecs(s,k)=bvd(BUN[s][k][1]).
- PLAN: teacher-forced cf states (doses scaled from h12, readings recorded) -> target = structural direct effect given readings - model change -> ridge readout on q -> dead value unit; calibrate via dose_edit.

## Session 7 (runs 179-202)
- Exact cf teacher-forced data: dose x m from h0 (run_cf edits) + readings shifted by true S_t=K*(dDose) (global SHIFT dict, mod_events rebound to mod_events_new).
  M (68668 rows: drug none/insulin/norepinephrine, h0, m, event, hour, S, mean_cf, mean_f, obs, b), WC (with_code mean over draws), LG (logits 4 draws).
- BUG FOUND: run_cf's states BEFORE its restart index are wrong (H off ~3%) -> factual rows must use run_cf(s,{},BUN[s][1][0],Xs).
  Fixed: replica factual NLL 1.5077 vs tool 1.5116; pred-mean err ~1.4 at all hours.
- Teacher-forced on exact cf histories the model moves only ~13% of truth: ins x2 h12 GF -1.64 vs -12.85; NE x2 MA +1.34 vs +4.18.
- solve_theta(lg,ev,v,T_SHIFT)->tilt; tilt_mean returns MEAN. vG[1:]=(cenGF-160)/25, vM[1:]=(cenMA-75)/8 (bin0 = 0).
- Plan: ridge readout theta~WC (glucose rows -> insulin unit, MAP rows -> NE unit), gated dead units 89/170 in value_bin.0, const cancelled in embed_transition.

## Session 8 (runs 208-237)
- frontier: expF = factual dNLL per glucose reading (oof). wf10: +0.099 (A=62% of S); wf1000: +0.021 (A=22%). Head readout too costly.
- Best linear readout on WC (Fisher dir, ins x2 h12 glucose rows): oof SNR 1.4, in-sample 1.6 (need ~4). => head-only edit limited (~1/3 of gap).
- embed.numeric dose cols x4: no effect (shift -0.86->-0.83). u+s*(u(2d)-u(d)): shift -0.52/-0.63 (worse). Model ~ignores u.
- z: POSTERIOR COLLAPSE in core: step output indep of z (z+10 -> diff 5e-7). z encodes GF surprise (CV R2 .70) but no effect on state.
- RTs[s] cols: minute,b(bundle idx),prow,event,observed,predicted_mean,nll. CEN[code] 16 centres = logit bins 1..16. fb_(code,x) fine bin.
- amp_eval(lam,stays)->[(s,k0,k1,lgf,lgc)] factual vs ins x2 teacher-forced (readings recorded); amp_stats/summ in /work/amp.py.
- DON'T reuse global names F (torch functional!), C, W, M. bvd/zpost restored to bvd_orig/zpost_orig.

## KEY (run 240-242): M cf data BUG - shifted readings never reached replica state (obs col shifted, states not).
- Replica test (30 stays): readings shifted -13 from h13 (core+head last_val; RT age is SECONDS) -> GF pred -11.1; doses x2 alone -0.90; both -12.07 (additive).
- => model persistence ~0.85; problem = direct (one-step) dose effect too small (-0.9/step vs needed ~-2..-3).
  Needed per-step direct kernel K_D(k)=KI(k)-0.85*KI(k-1): [-2,-0.7,-0.12,+0.1,+0.17..] per unit; sum -1.9. Edit needed is SMALL (~-1.5 mg/dL/step).
- Plan: Fisher readout w from WC (cf doses x2 readings recorded vs factual), head unit tilt with gain g; calibrate g via tool dose_edit.

## Session 9 (runs 245-251)
- Tool rollout (orig) ~1.5-1.9x teacher-forced direct (M) for ins; NE rollout < TF direct. Effective rollout feedback rho~0.4 (not .85).
- Ridge readout on WC (glucose rows, per (code,lastbin) centering, clip tilt targets): ins rho.4 lam30 oof frac .46 cost g1 .0018/glucose row, full .0107; NE rho.3 lam300 frac .31 cost full .0056/MAP row. Drug-bundle gating useless (88% rows).
- codes: GF6 GL10 MA7 MC5 ins8 NE9. EDIT v1: unit89 (ins->GF/GL, vG tilt), unit170 (NE->MA/MC, vM), gate via embed.code min-var, linear GELU regime, centering in embed_transition rows; gains 1.5/1.5. fns fit_full, build_unit, apply_units.
- runs 252-257: readout must NOT be projected off LN cols 0,3,6 (cost x50); project only drug LN cols [1,2,4,5,7] (Pd). fitw2(fa,c,v,rho,lam,0,Pd,1e-2): ins frac .46 cost g1 .0017 full .0073; NE frac .31 g1 .0009 full .0056.
- near-dead head units: 89,170,118,158,149. EDIT v2 (run 258): pairs main+cancel (89/149 ins, 170/118 NE), gate G=10 (min-var, orth to bins & LN), kap=1/sd, gains 1.5/1.5. fns unit_pair, apply_pairs.
- fwd5 factual both sel1: orig 1.51162 -> v2 1.51604 (+.0044; pred in-sample .0039). by event dNLL: GF +.0034, GL +.0247(!), MA -.0007, MC +.0039.
- fwd6 dose_edit both sel1, v2 g1.5: ins x2 -2.75->-3.49, half 1.38->1.94, NE x2 .69->.75, half -.58->-.64. => readout fit on cf (doses scaled, readings recorded) does NOT transfer to rollouts (likely reads 'surprise'). Need exact-cf (readings shifted by true S) training states.

## Session 10 (runs 264-285): CORE DOSE CHANNEL
- Arch: doses enter core ONLY via action token u_k -> prelude.0 attention. Prelude head 2 attends u with weight 1.00 on drug bundles.
- slot residual norm ~800, head2 output ~0.7 -> dose signal swamped (why model ignores dose). E=prelude_norm(slots) cov rank-deficient (102 null dims).
- Edit design: (1) kv rows 288:336 (head2 value) += cp*(pEI x wI2 + pEN x wN2), cp=40, wI2/wN2 = rate probes on ln1(u) (PR2), pEI/pEN = Ve[:,-1/-2];
  (2) core.b_in.weight += tI x rI + tN x rN (rI,rN null-space readers of dE; tI,tN = TC@V2_[:,-1/-2] scaled to SNR 3/6). Vars: WkvF, BinF, KV, BI.
- Replica disturbance at this scale: dnll all +0.0008 (GL +.002, MA +.002). Channel is one-step (decays within ~1h).
- Readout: J from factual edit-vs-orig is wrong (accumulated effects); use cf contrast (x2 vs factual, edited) for J.
- HT(n)=load(n): replica heads read saved edits. heads_full returns (wc (n,4,192), lg (n,4,17)).

## Session 11 (run 286+)
- Truth sel1: A x2 -13.95 x.5 +6.98; B x2 +4.28 x.5 -2.14. Orig rollout -2.75 1.38 .69 -.58.
- Dropped v2 head edits (cost .0044 too risky). Testing core channel only: mk(gI,gN) = kv rows 288:336 += pk (cp), b_in += gI*OI + gN*ON. Saved mk(3,-3) for tool calibration on sel1.
- fwd7 (75 sel1 stays): channel mk(3,-3) ~no effect (A x2 -2.65->-2.57, B x2 .745->.787). Channel abandoned, dropped.
- Arch: u_k enters state only as prelude token (z path unused). Test: scale prelude head-2 value rows 288:336 by k=4.
- fwd8/9: head2 value x4 (uniform): A x2 -2.65->-4.15, half 1.27->1.97, B x2 .745->.858, half -.435->-.619; factual +0.0011 (GF+.0049 GL+.0044 MA-.0014 MC-.0026).
- run292: targeted: amplify head2 value only along top dirs RA/RB of d ln1(u) under dose x2/x.5 (orth to mean u and LN bias). mk2(kA,kB). Test mk2(12,12).

## Session 12 (run 294+): 7 py runs left. Run294: inspect current kv edit, summarize fwd_8..11 (10/11 likely mk2(12,12) tests).
- fwd10/11/12 = mk2(12,12) on sel1 (75 tuning): A2 -2.65->-9.18, Ah 1.27->3.31, B2 .745->1.80, Bh -.435->-1.46. factual +0.017 (GF+.035 GL+.053 MA+.003 MC-.006) -> too costly.
- est event fractions GF.353 GL.100 MA.302 MC.245. linear: A2=-2.05-.594kA, Ah=1.09+.185kA, B2=.649+.096kB, Bh=-.34-.093kB. G cost ~1.46e-4*(kA-1)^2 (overall units).
- mk2(kA,kB) SAVES the edit itself. Tool deterministic per stay set. Run295: mk2(5.5,18) -> test dose_edit+factual on sel1.
- fwd13 factual mk2(5.5,18) sel1: +0.00422 (GF+.0144 GL+.0142 MA-.0029 MC-.0058); n GF1438 GL390 MA1217 MC976 (of 4021).
  G cost ~LINEAR in kA (~.0017/unit, not quadratic); M fit IMPROVES with kB. Additive fit: cost=.00176(kA-1)-.000217(kB-1).
  Targets: Bh truth reached at kB~19, B2 needs ~38 -> kB~22-28 balances. tuning ids = 2020000000+0..399.
- Run296: mk2(5,24) -> test factual + dose_edit on sel1.
- fwd14 factual mk2(5,24) sel1: +0.00461 (GF+.0151 GL+.0131 MA-.0035 MC-.0041): kB>18 adds GF cost, M benefit flat -> kB~14-16 best.
  3-pt fit: cost ~ .00117(kA-1)+5.3e-5(kA-1)^2 -.00036(kB-1)+1.4e-5(kB-1)^2 (sel1). B-part min ~-.0023 at kB~14.
- Run297: mk2(4.5,15) (pred cost ~.0022-.0029). Plan: factual sel1 + validate on other 325 tuning stays; dose_edit sel1. Keep >=1 py run reserve.
- fwd15 mk2(4.5,15) sel1: +0.00256 (GF+.0103 GL+.0096 MA-.0029 MC-.0048). fwd16 SAME edit on 100 non-sel1 tuning stays (batch1): +0.0100 (GF+.0158 GL+.0086 MA+.0090 MC+.0064) -> FAILS. sel1 unrepresentative (cheap)!
- non-sel1 tuning batches (offsets from 2020000000, excluding sel1): b1 1..133, b2 134..255, b3 256..374, b4 375..399.
- Run298: fallback mk2(2.2,5) saved + per-stay cost split by drug presence (fwd15/16). Run299 = final candidate, run300 = reserve.
- FALLBACK mk2(2.2,5) factual: b1 +.00179 (GF+.003 GL+.0052 MA+.0014 MC0), b2 +.00096, b3 -.00027 -> pop ~+.0008. (run298 split analysis failed: event col not found).
- b1 cost super-linear: G~.00341lA+.00213lA^2 (lA=(kA-1)/3.5), M~-.00017lB+.00463lB^2 (lB=(kB-1)/14).
- Run299: FINAL candidate mk2(2.5,8): pred b1 ~.0029, pop ~.0013; pred A2 -3.5 Ah 1.5 B2 1.4 Bh -1.1 (orig -2.65 1.27 .745 -.435; truth sel1 -13.95 6.98 4.28 -2.14).
  Validate b1-b4 + sel1 + dose_edit sel1. If any batch >.0035 or mean >.002 -> run300 = mk2(2.2,5) revert (last run!).

## FINAL (run 300): KEEP mk2(2.5,8) - kv rows 288:336 (prelude.0 head-2 value) amplified only along dose-change dirs of ln1(u): A dirs x2.5, B dirs x8.
- factual, all 400 tuning stays: +0.0012/reading (b1 +.0029, b2 +.0018, b3 -.0002, b4+sel1 +.0002) -> well under 0.005.
- dose_edit sel1: A2 -2.65->-3.70, Ah 1.27->1.71, B2 .745->.948, Bh -.435->-.591 (record-based truth est -13.95, +6.98, +4.28, -2.14).
- Lessons: sel1 (selected drug stays) understated cost ~4x vs general stays; cost super-linear in gains; B effect convex in kB (weak at 8, strong at 12).
- Remaining limit: true effects ~5x model's; reaching them costs far more factual NLL than allowed with this channel.
