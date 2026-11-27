# Notebook
## Setup (run 1-4)
- Model: event-stream world model, actions (insulin, norepi) enter via encode_action -> u_k, used in z_post and prelude token.
- Records: readings at integer hours; infusion records start at hour+0.0833, run 1h. Dose chosen after hour-t readings precedes hour t+1 readings.
- Clinicians chose doses (confounded).

## Data facts (runs 3-19)
- All stays 25.17h; readings at integer hours -1..24; infusions at h+5min for h=0..24, 1h each.
- finger vs lab glucose: same quantity (paired diff -0.02, sd 9). arterial MAP ~3 mmHg below cuff (sd 6.7).
- INSULIN POLICY: off if last glucose<140, on if >160, random (hysteresis) 140-160. Rate = base(band)*mult noise (CV .33), round 0.5:
  base 1 (<180), 2 (180-220), 3 (220-260), ~4.3 (>260). Noise ~iid (no stay persistence, uncorrelated w/ lab@t, past glucose).
- INSULIN EFFECT (within-band dose noise): next-hr finger glucose -2.0/unit/h, linear across bands & rates 0.5-7.
  lag response to dose shock at t: k=1..6: -2.01,-2.25,-1.81,-1.30,-0.73,-0.45 ; dose feedback on later doses only -0.03..-0.05.
- NOREPI POLICY: titration on log rate, uses arterial MAP if present at t else cuff else last; <=64 up ~+0.15, 66-74 hold, >=76 down ~-0.25,
  noise sd ~0.3 log; capped at 1.0 (13% of records at cap), step 0.01.
- NOREPI EFFECT (crude shock, maybe biased): MAP resp k=1..7: 3.3,5.4,6.4,6.5,6.0,5.6,4.9 per +1 mcg/kg/min shock; shock persists in rate .85,.73,.64...
  crude deconvolved impulse: 3.3,2.6,1.8,0.9,0.4,0.3 (~9 mmHg per sustained unit).

## Effect estimates (runs 20-26): cross-fitted double-ML with GBM on history feats, deconvolved by first stage
- INSULIN -> glucose impulse h_k (k=1..12, per unit/h for 1h): -1.99,-2.39,-2.11,-1.72,-1.22,-0.95,-0.69,-0.55,-0.35,-0.15,-0.12,-0.05
  ~ -1.99*k*0.6^(k-1); sum ~ -12.3 per sustained unit/h. Homogeneous across subgroups (ne on/off, art, time, glucose band). Linear in dose.
- NOREPI -> MAP impulse h_k: 3.35,2.60,1.75,1.03,0.42,0.27, ~0 after (sum ~9.6 per sustained 1 mcg/kg/min).
  Heterogeneity: lag-1 effect lower (~2.5-3.4) when MAP<60; ~3.6-4 otherwise; no clear rate saturation in other strata.
- Cross effects ~0 (insulin->MAP 0; norepi->glucose +0.6+-0.25 lag1 only).
- Tuning: 217 stays insulin>=12, 153 norepi>=12, 87 both, 96 norepi>=12 w/ arterial line.

## Model vs truth (fwd 1, original, 100 tuning stays: 87 both drugs + 13 norepi/art)
- readout rows = stay-hours 13-24 with the readout reading recorded, 'reached' = hour > first dose hour (>=12).
- model: ins x2 -2.15, ins half +1.23, ne x2 +0.34, ne half -0.06
- my truth (impulse h applied to recorded doses): ins x2 -13.8, half +6.9, ne x2 +4.6, half -2.3
- model direct (lag-1) sensitivity: insulin -0.35/unit (x2), -0.62 (half); norepi +0.11/unit. Truth: -2.0 and +3.35.
- data conditional kernel (DML, holding 4h readings): insulin c_l = -2.01,-1.45,-0.47,+0.22,+0.73,+1.09 ; norepi 3.30,0.99,-0.25,-0.99,-1.09,-1.45
  (positive late terms = baseline-correction / partly truncation confounding) -> calibrate empirically instead.
- fwd 2: states (original) on 99 train stays (sel_states.json).

## Architecture facts (run 40-41)
- value_bin head: Linear192->192, act, Linear192->17 (shared by all reading codes); embed_transition.value_bin (13*17 x 17) bias by code x last bin.
- z_post.0 (192x576) on [ctx,e,u]; dose reaches heads only via u_k -> z_post/prelude -> state -> view -> q.
- Plan: code-gated linear correction term in value head hidden layer, reading a dose direction in q.

## Reconstruction findings (runs 42-67)
- ALL MLPs are GELU (not ReLU). LayerNorm eps 1e-5. type_emb row 1 for value atoms. gap input log(dt_s)/10. Time flow exp(-dt_s*exp(flow.a)) applies for heads view.
- z_post order [ctx,e,u]; unit table rows: glucose 4, MAP 2; my posterior recon corr ~0.85 (not exact).
- value head recon: last_numeric layout fitted (A2, 8x4 on [1,has,age,zlast]); held-out MAE 1.3 vs tool predicted means.
- fwd3: factual 3 stays (mistake). fwd4: factual original on 99 states stays (value NLL 1.520).
- State after insulin bundle encodes dose residual (dose - band-expected) with CV R2 ~0.42 (out/slotmean/h).
- Model factual error regressed on dose shocks: insulin lag1 -1.47(.29), lag2 -1.43(.30), lag3-4 ~0 ; norepi lags1-4 2.5,2.5,1.4,2.5 (+-0.9) (shock persists in rates).
- PLAN: repurpose never-active value-head hidden units (GELU) as gated linear units reading a probe of q for dose-minus-readings-implied-dose; calibrate gains with dose_edit.

## Restart after run 73
- run 73: no dead value-head units (most negative max pre -1.99 unit 35; 53 -1.84, 125/24 -1.66, 180 -1.47).
- gate direction: glucose readings proj 1.16-1.26 (codes 6,10 emb 1.44), MAP -1.46..-1.57 (codes 5,7 emb -1.28), other codes 8,9,11,12 in [-0.33,0.21].
- idea: paired units sharing the gate (one reads the probe) so gate variation cancels; needs cost check of repurposing.

## Restart 2 (after run 121; runs 74-121 were NOT logged - recovered from kernel vars)
- fwd5-7 = states(original) for new300.json stays (3x100) -> S5,S6,S7 / D5,D6,D7. errs[300]: some recon check, mean err .056.
- Kernel arrays for 99 sel_states stays (5521 readings): Q99 (q), WC99 (with_code), PRE (value-head pre-acts), TR99 (transition bias), CENm (centres), pmall; eps_ins/eps_ne (30000x25) dose residuals per stay-hour.
- No function source retrievable; rebuild key pieces explicitly.
- PLAN: paired GELU units in value head (A = B + eps*probe, W2 cols +-lam*tilt; gate by code-direction), removed units' mean absorbed into embed_transition.value_bin; probe on q for dose-shock part of factual error; calibrate gains via dose_edit per-hour kernels. Everything analytic/regression-based, no optimiser over weights.
- run130: with_code VERIFIED = Q99 + E_code + last_numeric(A2@[1,has,log1p(age_h)/10,zlast]) + E_val[lastbin], bins side=left; TR exact. WCA/TRA/lbA/obA in kernel. means MAE vs tool 1.26 (z-draw noise). recon NLL 1.528 vs tool 1.520.
- run131: unit repurpose cost (act -> per (code,lastbin) mean, absorbed in embed_transition) tiny: cheapest 18,59,20,40,186,12,61,100 (~-0.0007), median 4e-5.
- run133-135: model factual err ~ resid lags: INS beta (-1.72,-1.66,-.43,-.44); NE (2.7,2.7,1.5,2.5). Ridge probe on with_code: R2 lag1 .21 INS, .10 NE; unbiased GLS decoder hopeless (dose dir collinear w/ readings dirs).
  probe-in-value-head route: factual cost too high (INS lam2 .0034, lam3 .0085; NE lam5 .004) for tiny cf response (-0.44/unit at lam1). -> REJECTED.
- NEW PLAN (run138): clean dose channel via quiet z dims (18 sd.046, 28 sd.058, 27): repurpose 1 z_post hidden unit per drug: pre=p.u + B (B big, linear GELU regime; u=0 for no-drug bundles -> exactly 0), z_post.2 row 18 col j = lam, bias -= lam*B.
  Then measure delta-q via tool states 'both', build value-head paired-unit readout + last-bin table compensation; calibrate with dose_edit.
- run140: q rebuild exact (view_from(prev_row=S2 row k, dt), z=prior mean) == Q99. u_k rebuild: setenc_my(drug_atoms(i,t,'both',mult)) (empty->0); z_post recon err .05 (ctx-side).
- run144-145: linear decoders of rate from u_k (no intercept, ridge a=.01, trained on dose-edit-augmented encodings): INS |p|379 ratios ~1 at x0..x3; NE |p|142. saved /work/p_ins.npy p_ne.npy
- run147: WRITER EDIT SAVED (z_post.0/.2 w,b): repurposed dead z_post units 55 (ins) & 109 (ne): pre=5+p.u, out col -> z18 (lam .3/unit/h) / z28 (lam 2 per mcg/kg/min), bias -lam*gelu(5). offline: dz18=.29*rate, dz28=1.83*rate, others 0.
  NEXT: fwd states both on sel_states (99) -> measure delta-q at readings; then value-head paired-unit readout.

## Restart 3 (after run 149)
- fwd8 = states BOTH (writer edit z_post 55->z18 ins, 109->z28 ne) on sel_states 99 train stays.
- ARCH recap: state update uses z POSTERIOR MEAN; heads use z drawn from PRIOR (4 draws, probs averaged). value head = 17-class softmax (0 none, 1..16 fine bins), mean=sum p*centre.
- run150-167: WRITER BUG FOUND. (1) empty-action u = U_EMPTY (not 0): p_ins.U_EMPTY=-19.2, p_ne.U_EMPTY=-0.51 -> fwd8 had const dz18=-1.5, dz28=-1.02 on all non-drug bundles.
  (2) TRUE u = encode_action over START atoms only (END atoms excluded although n_atoms counts them); drug atoms include categorical rows: insulin unit 3, rate_unit 2, route 1; NE unit 4, rate_unit 3, route 1; action 0, status 0. bins = 1+#edges<x (strict). z clip +-10 (insulin rate>11.5 saturates).
  corrected recon u_rec(ri,rn) matches tool-implied projections: rmseI .047 (|p|379), rmseN .012. Functions datom(), u_rec(), u_c() in kernel.

## Restart 4 (after run 183)
- runs 168-183 not logged; run 184 = inventory of kernel state + notes head.
- Design principle: correction = sum_l kappa_l * dose_innovation_{t+1-l} (innovation = dose - policy-expected dose given readings/prev dose),
  kappa from regressing model factual error on innovations; written via clean z channel; then rollout handles propagation. Verify with dose_edit.
- run 189-190: insulin rate linearly decodable from ctx at drug bundle (cvR2 .71, resid .65 vs policy .62). NE: ctx R2 .59 vs policy-features R2 .87 (innov sd .128); prev NE rate in ctx only R2 .62.
- run 191-198: fwd8 z18/z28 writes barely reach heads: dctx .04, dout .03 (natural 1.55); dh ~1.4 per slot common (natural 76). GLS SNR of dq on lagged doses per unit^2: NE lag1 ~1.2 (gluc rows), .5 (map); insulin ~.01. => z-channel as built is USELESS (signal buried). view_from(prev,dt)[0] = q exactly; R.k/R.dt index fwd2/fwd8 rows.
- every drug bundle has START atoms for both drugs (rate 0 when off); dose edit keeps atoms, scales rate&dose (x0 -> rate 0).

## Restart 5 (after run 202)
- run 202 failed (means_from shape bug, 64 vs 17). Writer edit (z_post 55/109) still saved but judged useless. Re-reading ARCHITECTURE.md (state update / slots / decay) to design a stronger dose pathway.

- runs 205-233: c_out not identity, abar~.35: no linear SSM channel. Drug bundles at h+0.083 (action-only); END atoms are encoded in e (attrs none). Readings never in drug bundles.
- vG/vM = z-surprise directions (corr .76/.58 with glucose/MAP surprise); next-pred gain .54/.66.
- ctx decoders of rate: ins cvR2 .70 (innov sd .66), ne .58 (sd .23) saved /work/decoders.npz.
- model factual err ~ ctx-innovation lags: gluc_finger -2.04,-1.43,~0; map_art +4.05,+.76.
- dead z_post units 12,88,105 (max pre <-4.4). WRITER v2: j0=12 gate, jA=88 ins innov, jB=105 ne innov (u-decoders pg,pA,pB fitted on realistic+edited rates, lam1e-4), dz=lam*innov*dir. saved lamA=-.18 lamB=.6.

## Restart 6 (runs 272-279)
- fwd9 factual both (writer v2): no change; fwd10 states both: z write ~.03, h diff .03 vs natural 10.6 -> z-token route dead (prelude LN). fwd11 dose_edit both tuning: writer v2 ~no effect (ins x2 -2.07 vs -2.15).
- a_bar .29-.43 all dims: no slow linear memory. DROPPED all writer edits (run 278).
- NEW APPROACH: amplify the drug-rate-induced deviation of the action encoder output u (model own dose pathway) per drug: encode_action.mlp2.2 += (k-1)*ridge map hid->deviation; deviation in LN-normalised coords (x sd) relative to ref rates (tuning means ins 1.70, ne 0.49); empty bundle pinned 0. Fit R2~1. set_amp2(kI,kN) in kernel. T must stay the tensor dict (helpers use it); TT = converter.
- tuning.csv: drug rows only when running (no rate 0); ins rate median 1.5 p95 3.5; ne median .45 p95 1.0.
- run279: saved kI=5,kN=8 -> next: dose_edit both + factual both to calibrate.
- run279: u-amplification k=(5,8): insulin u already saturated (x2 |du| 8.2 vs natural spread 8.8; amp ratio only 1.6), factual distortion 5.2 -> REJECTED. NE x2 |du| .49 (amp ratio ~5).
  => insulin bottleneck is DOWNSTREAM (model sees dose, barely maps to glucose).
- run280: EMULATION edit: encoder last layer maps hid(r)->s(2r) both drugs (ridge, exact-ish) to get direct dq response at readings via states both on sel_states 99. Plan: GLS readout dirs in q, paired gated value-head units (insulin->glucose codes, NE->MAP codes), calibrate with dose_edit.
- run283: KERNEL OOM RESTART (all vars/helpers lost). run284: dropped emulation edit (model = original). Lesson: never rebind single-letter globals; read train.csv in chunks.
- run284 (out-space GLS from fwd12 emulation): INSULIN dose barely in out at glucose readings: SNR .11/unit^2 (noise 3 units) -> insulin readout NOT viable; info lost in drug-bundle step (prelude). NE strongly in out at MAP readings: SNR 14.8/unit^2 (noise .26) but value head ignores it.
  Readings bundles at integer hours, drug bundles at h+.083, so the next reading follows the drug bundle directly.
- run285: INPUT-LEVEL dose amplification (shared): embed.numeric cols 2,4 (z_dose,z_rate) x k, bins.rate/dose rows 1..16 expanded around row 8 (median bin for both drugs) by k. set_inamp(k). Testing k=6 (dose_edit tuning + factual).

## Restart 7 (run 286)
- fwd13 dose_edit both tuning100, input-amp k=6: ins x2 -1.70 (orig -2.15), half 1.43 (orig 1.23); ne x2 .10 (orig .34), half -.105 (orig -.057) -> input amp USELESS, REVERTED (no edits now).
- run287: ARCH re-read: dose reaches value head only via core state (prelude tokens [h+slots,e,u,z_in(z)] -> E -> blocks/b_in -> coda -> out_norm); view=out+out_norm(mean h_flow)+gap+z_head(z_prior draw); q=query_norm(view+type_emb); with_code adds per-code registers only.
- run288: TARGETS (tuning100): ins x2 -13.8, half +6.9; ne x2 +4.6, half -2.3. orig -2.15/+1.23/+.34/-.06.
  WRITER v3 saved: z_post.0 unit 88 pre=8+pI.u (ins), 105 pre=8+40*pN.u (ne), 12 pre=-8-pI.u (on only at U_EMPTY, ~11.2); z_post.2 rows 18/28 := rate_I / rate_N (+nat mean) with unit-12 compensation; z_in cols 18/28 x FI,FN (0.5/2.0 normalized-token per unit rate), bias compensated. NEXT: states both on sel_states -> check z18/z28 = rates, measure dq at readings; then value-head reader (rate minus E[rate|last bin] via embed_transition).
- run289-290: z route dead (z token sd .18; prior writes already saturating). q rebuilt from emulation states: dq per unit rate |b| ins .028 (eta_nat .06, contaminated by NE 9x) -> INSULIN READER IMPOSSIBLE; NE lag1 |b| .286, perp-reader eta_nat 5.6, cost est ~.003 at kappa 1.5.
- run291: NE READER SAVED in attr_heads.value_bin: paired units 120 (gate+readout) / 110 (gate), gate on MAP codes 5,7 only (s=22.5), readout w (perp registers, w.bN1=1), tilt lam*(centre-70), kappa=1.5/VarP35; old units' per-code mean absorbed in embed_transition. NEXT: dose_edit+factual both tuning100, rescale lam.

## Restart 8 (run 292)
- run292: tuning100 ids (= fwd13 stays) printed; helper _set_kappa_ne(k) rescales value-head out cols 120/110 (saved kappa 1.5). NEXT: fwd14 dose_edit both tuning100, fwd15 factual both tuning100, rescale kappa toward NE x2 target +4.6 keeping dNLL<.003. Insulin left at original (no viable pathway).
- fwd14 dose_edit both tuning100 (NE reader kappa1.5): ins x2 -2.12 (orig -2.15), half 1.25; NE x2 1.46 (orig .34), half -.225 (orig -.057): reader works but asymmetric (half only 15% of x2 contribution; model's own NE response equally convex).
- fwd15 factual both tuning100: value NLL 1.4852 -> 1.5191 (+.034, limit .005!): map_art +.044, map_cuff +.083, glucose 0. run293: offline decomposition of cost (offset vs noise) from fwd15 csvs.
- run293 (fwd15 csv analysis): NE reader shift d: art mean +.19 sd 1.80 corr(d,res) .09; cuff mean -1.01 sd 3.46 corr .005. Cost ~ Var(d)/(2V) (V art~28, cuff~67): it is reader NOISE, not offset; cuff gets 2.4x larger shift per reader unit (shared tilt, wider predictive). Recentred cost pred per value: f.25 .0005, f.5 .003, f1 .015. Reader SNR in rollout (1.1 mmHg x2 vs 1.8 noise) ~.6 -> full NE target infeasible.
- run294: kappa -> 0.5 (f=1/3), no recentring. Expect factual +~.001, NE x2 ~+.7, half ~-.11. NEXT: fwd16 factual both, fwd17 dose_edit both (tuning100).
- fwd16 factual both kappa.5: +.0029 (art +.0047, cuff +.0062). fwd17 dose_edit kappa.5: NE x2 .698 (contrib +.362, linear in kappa), half -.126, insulin unchanged. Cost fit ~ .0376 f^2 - .0037 f (f=kappa/1.5).
- run295: kappa -> 0.4 (pred cost .0017, NE x2 ~.63). NEXT: factual both on fresh tuning set B (first 100 tuning ids not in A) to verify cost.
- fwd18 factual both set B (100 fresh tuning ids 2020000001..146 not in A) kappa.4: +.0035 (cuff +.0087 n1765, art +.0005 n896, glucose 0): cuff-heavy set -> cost higher. Pooled est kappa.4 ~.0026, cuff-heavy worst ~.004 -> too close to .005.
- run296: kappa -> 0.3 (expected cost ~.0015-.002; NE x2 ~.55, half ~-.10). NEXT: factual both on A and B, final dose_edit A, then FINAL.
- run296: setB kappa.4 cuff d mean -.50 sd .90 (meanres +.48) -> systematic cuff offset (A: -.27 equiv), arterial ~+.05. kappa set 0.3.
- run297: cuff recentring via ET rows 17*cuff..+16: logits bins1-16 += (0.30/60)*(centre-70) (constant tilt, cancels in dose-edit diffs). NEXT: factual both A,B; dose_edit A; FINAL.
- fwd19 factual A (kappa.3 + cuff recentring): +.0019 (art +.0017, cuff +.0061, glucose 0). fwd20 factual B: +.0008 (cuff +.0020, art -.0002). pooled ~+.0014 << .005.
- fwd21 dose_edit A final: ins x2 -2.137 (orig -2.148), half 1.213; NE x2 +.582 (orig .336), half -.079 (orig -.057).
## FINAL STATE (run 298)
- Edits: value-head NE reader (units 120/110 paired, gated on MAP codes 5 cuff/7 arterial, kappa 0.3), ET compensation for repurposed units (blocks 5,6,7,10) + cuff recentring tilt.
- Beliefs: true effects (from records) ins x2 ~-14 / half ~+7 mg/dL; NE x2 ~+4.6 / half ~-2.3 mmHg. Model state carries almost no insulin-innovation info at glucose readings -> insulin left at original (-2.1). NE info present but noisy (SNR~.6) -> only small kappa affordable under the .005 factual limit; NE response convex (decreases weak).
