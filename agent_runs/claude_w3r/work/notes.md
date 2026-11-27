# Notes

## Session 1 (restart 1)
- Step 1: reading ARCHITECTURE.md and data README.
- Architecture: dose/rate enter only via insulin/norepi atom embedding (embed.numeric cols z_dose,z_rate; embed.bins.dose/rate),
  -> encode_action -> u_k -> z_post and prelude tokens. Values predicted via attr_heads.value_bin + embed_transition.value_bin[code*17+lastbin].
  Dose edit: doses x2 / x0.5 from hour 12, readings from h13 drawn by model; readout = mean pred glucose_finger (or map_arterial) edited-factual over h13-24.
- DATA: readings hourly (int hours -1..24), infusions start t+5min, 1h long. Weight fixed 80kg -> dose∝rate. All stays 25.17h.
- INSULIN POLICY: banded sliding scale on last glucose_finger: <150 off; 150-180 base1; 180-220 base2; 220-260 base3; 260-300 base4; >=300 ~6.
  Within-band rate = multiplicative noise (log-sd ~0.35), i.i.d. across hours (lag1 corr 0.008) -> as-if random.
- RESULT: one-hour effect of +1 u/h insulin deviation on next-hour glucose_finger: -1.98 (se .05) mg/dL; placebo on g_{t-1}: -0.02 (se .05).
- Reduced-form IRF of insulin deviation x_t on g_{t+s}, s=1..8: -1.97,-2.23,-1.74,-1.19,-0.79,-0.49,-0.24,-0.12.
  Policy feedback dr(m)=effect of x_t on r_{t+m}: 1.01, -.046,-.061,-.052,-.034,-.018,-.015,-.004.
  Recursive structural IRF h(k) (fixed doses), per 1 u/h for 1h: -1.95,-2.29,-1.94,-1.51,-1.15,-0.84,-0.55,-0.38 (k=1..8). Peak lag2, decay ~0.75/h.
  (2SLS distributed-lag gave positive lag7-8 -> likely misspecified; distrust.)
  Rough expectation: insulin x2 readout ~ -10 to -15 mg/dL (to be computed on actual dose patterns).
- FWD1 (original, tuning sel1 100 random stays): insulin_x2 -0.53, insulin_half +0.09, ne_x2 -1.33, ne_half -0.48 (n stays 55/21).
  Readout rows = recorded outcome readings (gf / map_arterial) at hours 13-24; reached = a drug record started in [12,hour). Verified.
  Records-based linear-IRF estimate on same rows: insulin x2 -12.5, x1/2 +6.25 -> model underestimates ~20x.
- KEY DESIGN ISSUE: dose ~ band(glucose); model's glucose dynamics absorb band-level insulin effect. Adding effect ∝ actual r would
  double count -> factual NLL disaster. Need edit acting on DEVIATION r - E[r|history] (z posterior = 'surprise' may encode it).
- NOREPI POLICY: titration AR-like: r_t ~ 0.27+0.74 r_{t-1}+0.04 r_{t-2}-0.003 MAP; start if MAP<65; stop sometimes MAP>=75; cap 1.0.
  Simple cell-mean policy failed placebo (x vs m_{t-1}: -1.73). Cross-fitted HGB policy on (m0,m1,m2,r1,r2,r3,mc0,t): R2 .93, innov sd .092;
  placebos m3,m4 pass (-0.09±.10, 0.07±.10).
- NOREPI EFFECT on map_arterial, structural IRF per 1 mcg/kg/min for 1h, lags1..12:
  3.45,2.79,1.79,1.09,0.50,0.22,0.26,-0.14,0.07,-0.11,0.20,-0.18; cum ~10 mmHg (reached by lag 5-6).
  Model original: ne_x2 -1.33, ne_half -0.48 -> wrong sign/size.
- INSULIN refined (clean sample: gf measured at t, g>=155 so always on; band cell-mean deviation; IPW for meas at s>=2 (meas=1 if ins prev hour else 0.30 random)):
  struct IRF lags1..12: -2.0,-2.45,-2.1,-1.7,-1.3,-1.0,-0.7,-0.5,-0.35,-0.25,-0.25,-0.3 ; cum -12.6..-13.2 by lag12. Robust to controls & HGB-x.
  (full-sample HGB version gave positive tail -> contaminated by on/off region; distrust.)
  Lag-1 slope by band: -1.62(r~1),-1.99(r~2),-2.20(r~3),-1.75(r~4); quad term ~0 -> LINEAR in rate (log model rejected).
  Slope/G ~ -0.0098,-0.0101,-0.0094 -> effect maybe ∝ G*rate.
- VOCAB: 4 heads; z=(x-median)/iqr clip ±10; bins 1..16 = 1+#edges below x. gf centres 73..271 (bins ~10-17 wide mid-range).
  insulin rate med 1.5 iqr 1, edges top 5.0 (bin16 = >5). NOREPI rate med .46 iqr .6, edges top 1.0 -> doubled norepi >1.0 lands in bin16 & z>0.9:
  NEVER seen for norepi in training (OOD) -> explains odd ne_x2 answer. Only value NLL of glucose/MAP readings counts for factual fit.
- FWD2 states (orig, train selS 100 stays w/ both drugs+art line; /work/fwd_2). Probes (ridge, grouped CV):
  insulin rate at insulin bundle R2 ~0.65-0.69; band-mean c R2 0.90 (persists: .89,.86,.82 at later bundles);
  DEVIATION r-c: R2 0.24 at the insulin bundle, 0.02 one bundle later, ~0 after -> model FORGETS dose deviation immediately.
  => head-only edit can't give multi-hour effect; need core edit carrying dose memory. Plan: local re-implementation of model
  (validate vs states), save code in /work/impl.py.
- True-effect structure guess (insulin): AR(1) glucose rho~0.75 + direct effects b1=-2.0,b2=-0.95,b3=-0.26 reproduces IRF.

## Session 2 (restart 2) - local re-implementation attempts (runs 39-64)
- ARCH details confirmed: GELU (exact) everywhere; gap input = log(dt_seconds)/10; states out == core.out_norm(mean_slots(h)) exactly (h = post-coda y).
- Bundling (build_bundles) matches model exactly (times, n_atoms incl. END atoms).
- Embedding tables: row0 = padding (zero) for ekind, text, bins, categoricals; untrained rows norm~0.28 (kq1-3, cat row1s). Used: unit rows 2-5, rate_unit rows 2-3, kq[0].
- z_post reproduction: median abs err ~0.05 (|z|~0.58) at best (END atoms via OBS encoder better: 0.015 on action bundles). Unit mapping barely matters.
- STRIKING: model posterior z dims 9 & 20 at insulin bundles are BINARY in rate: rates 1.0/1.5 -> +1.08/+0.84; rates 0.5 and >=2 -> -1.1/-1.3 (sd .01).
  My implementation gives a weaker, smooth version (0.5 -> +0.3!). Could not find cause (bins, z, END handling, pooling, LN all tested). Insulin rates in data are multiples of 0.5; rate==dose.
- Insulin rate edges have duplicates [0.5 0.5 0.5 1 1 1 1 1.5 1.5 2 2.5 3 3.5 4 5] (side=left: 0.5->1, 1->4, 1.5->8, 2->10).
- DECISION: stop exact z replication. Pivot: implement HEADS locally from states (out,h) and validate vs factual predicted means; design head-level edit + empirically verified dose channel.
- KEY DESIGN (insulin): desired added term to glucose pred at reading s+1: b1(r_s - m(g_s)) + b2(r_{s-1} - m(g_{s-1})) [+b3...], m = band-mean policy.
  Compensation -m(g_last) can live in embed_transition.value_bin (per last-bin rows). Action bundle at s:05 holds r_s (start atom) and r_{s-1} (END atom) -> lag1+lag2 available in one bundle.
  Feedback via drawn glucose carries AR(1) rho~0.75; with b1=-1.95, b2~-0.83, b3~-0.22 reproduces structural IRF.

## Session 3 (restart 3)
- Run 77: local head impl vs factual pred mean (z=prior mean): median abs err 2.46 -> not exact yet.

## Session 4 (restart 4)
- Run 79: crosstab insulin rate vs categorical columns (looking for cause of binary z dims 9/20).
- Reminder: head z drawn from prior 4x at eval -> factual pred means are noisy; local head check must allow for this.

- Run 83-88: local head check. Register age must be measured at STATE time (minutes): log(1+ (t_state - t_last)*60)/10. typ=1, num[3]=1, last_status unused.
  Remaining error median ~2 mg/dL (ratio 1.4 vs 4-draw noise) -> local head approximately right, not exact. Use end-to-end calibration.
- Run 89 (gf rows of fwd_3, train selS): R2 from q: rate 0.69, dev(=r-bandmean) 0.41. MODEL partial slope of pred on dev = +0.31 (wrong sign: dose used as glucose proxy).
  Two-feature view: state holds clean band + noisy dose f=r+e (sd_e~0.9). Head term c*(f-m) with c=b1-gamma=-2.31 gives correct interventional slope,
  factual cost only ~+0.0004 nats/gf reading. Probe p=E[dev|q] has slope k~0.41 on true dev -> use T=(c/k)*p.
- Policy: insulin never in hours<=2; after that P(r=0|g>=155)=0.08.
- Run 90: value head (Linear192-GELU-Linear17, input with_code) has 17 DEAD hidden units (max act<0.05 over all 5684 rows).
  PLAN: gated linear channel with 2 dead units: A=GELU(G+P), B=GELU(G), W2[:,A]=u, W2[:,B]=-u -> adds u*P on gf rows (G>>0), 0 elsewhere (G<<0).
  u = tilt vector (c_b - cbar)/Vbar (u_0=0). Same for norepi->map_arterial with 2 more units.

- Run 92 (panel policy cells): insulin given ~only when a FRESH gf reading that hour: band means 0/1.03/2.07/3.11/4.15 (P on ~.99-1); not fresh -> P on ~.02.
  With proper policy mean, dev1 sd .56: probe R2 from with_code only 0.05 (slope k .076); dev2 R2 .01. Raw rate R2 .73 (via band).
  Model pm slope on dev1 = +1.2 (wrong sign). => HEAD-ONLY EDIT NOT VIABLE (needs 13x rescale -> big NLL hit).
  Need upstream channel carrying the insulin (and norepi) rate into the head-time state. Next: inspect core structure, dead units.

## Session 5 (restart 5) - local CORE STEP implemented (runs 95-117)
- Local core step works (validated vs fwd_2 states, starting from true prev state and model posterior z):
  time flow h*=exp(-dt_s*exp(flow.a)); prelude tokens [h+slots(4), e, u, z_in(z)] (7 tokens always, no mask);
  pre-norm layer (norm1->attn, norm3->ffn), E=prelude_norm(slot tokens); blocks x=h+E: per layer x+=selfattn(norm1 x); x+=cross(norm2 x, Mem+AE, Mem+AE)
  where Mem = e vectors of ALL bundles incl current (last 128), AE = core.age(log1p(age_seconds)) ADDED TO MEMORY before kv (both k and v);
  x+=ffn(norm3 x); hn=abar*h+sp*b_in(E)+(x-(h+E)); y=c_out(hn); y+=attn(coda.norm1 y); y+=ffn(coda.norm3 y); out=out_norm(mean_slots(y)).
  4 heads x 48, no attn biases. END atoms -> OBS encoder (e); start atoms -> encode_action (u). Unit rows: mg/dL=3, mmHg=4 (better), insulin 4->?, norepi 5.
  Accuracy: out median abs err 0.022, h rel err 0.045 (action bundles better than measurement bundles).
- h is HUGE (slot norm ~800, a few dims 100-200), slots nearly identical. LN scales ~58-65 everywhere.
- PRELUDE ATTENTION: head0, head3 -> slots uniformly; head1 -> e token 100%; head2 -> u token 100%; nobody attends z.
- Jacobians at insulin bundle: dXslot/du top sv 114 (then 4.5); dE/du top sv 1.48; dout/du 0.049 (rank-1); dout/dE <= 0.053; dout/dz ~0.
  => u reaches E fine (via head2) but E->out is attenuated (LN of huge y). z posterior has ~no effect on state.
- PLAN: find E-direction carrying insulin rate linearly (low normal variance), amplify E->h via rank-1 edit of core.b_in (or c_out),
  read at gf value head via dead units (value_bin.0 has 17 dead units) with tilt vector; compensate band mean -b1*m(g).
  Persistence of injected h-component (abar~.3-.44/step) may supply lag-2/3 terms; glucose feedback supplies the rest (avoid double counting).

## Session 6 (restart 6)
- Kernel fns have no source (exec'd); use as black boxes: stay_data(s)->(B,rows,EU) [B=(t,atoms), atom rate=(code,val)], step_parts(k) with globals B0,rows0,EU0 -> (h?,E,y,out,att[4x7x7]).
- Probe at drug-start bundles (100 train stays, 2406 bundles): E R2 ins .64, ne .49 (noisy: h dominates); u R2 ins .995, ne .984 -> rates clean in u.
- fwd_3 factual residuals (orig) on lags: gf r1..r3 -1.27,-.28,+.42; MAP r1..r3 +4.15,.01,-1.10 -> model MISSES drug effects factually.
  Gaussian dNLL (per reading, all readings): insulin deviation form b=(-2,-.95,-.26) with band comp: -0.0008 (good); insulin levels: +0.026 (bad).
  MAP levels w/o intercept bad (+0.02..0.08); WITH intercept (+mc,r3 free): -0.005 (good); fixed b1=3.45,b2=.72 + free intercept: -0.005. mc coef ~0.
- READOUT = pooled mean over reached stay-hours. Records-IRF targets on sel1 (tuning): ins x2 -13.71, half +6.86, ne x2 +5.17, half -2.58 (n 484/238).
- PLAN: channel u -> prelude head2 (rank-1 on attn.out) -> E -> b_in rank-1 -> h -> out -> value head dead units; comp: insulin band mean via register, norepi intercept.

## Session 7 (restart 7) - recovered state (runs 171-176)
- Runs 118-170 were NOT noted. Recovered: kernel fns (no source) use dict Wt (torch f64 weights) -> can simulate edits by
  temporarily modifying Wt[...]. Head fns use W (numpy). step_at(B,rows,EU,k,du,de)->core_step_v; stay_data(s)->(B,rows,EU);
  SX(5301,992)=fwd_2 states (SST stay, SHR hour). UX(9918,192)=u vectors (YXi/YXn rates); run170: linear probe of
  ins/ne rate from u holds under all 4 edits (slope~1, resid sd .03-.17) incl OOD ne x2.
- ARCH (re-read): new h = y = coda(c_out(h_mid)), h_mid = abar*h + sp*b_in(E) + (blocks(h+E)-(h+E)); out=out_norm(mean y).
  view = out + out_norm(mean_slots(h after time flow)) + gap + z_head(z). with_code = q + code emb + last_numeric[has,log(1+age)/10,z_last,z_last_rate]
  + bins.value[last bin] + bins.rate[last rate bin] + last_status. value logits = value_bin(with_code) + embed_transition[code*17+lastbin].
  Dose edit doubles rate&dose of EVERY insulin atom from h12 (incl END atoms). Readings drawn & fed back from h13.
- Timing: reading bundle s:00 (gf,map), action bundle s:05 (starts->u, END of s-1 infusion->e). Reading s+1:00 is predicted from
  state after s:05 bundle -> lag1 (r_s in u) and lag2 (r_{s-1} in END atom e) are both in the action bundle.
- PLAN A (3 stages): (1) prelude attn.out rank-1 per drug from head2 value (head2 attends u 100%), into a LOW-VARIANCE slot-token
  direction d (zero at no-action: v.V2(norm1(0))=0); (2) core.b_in rank-1 per drug reading that signal from E, writing direction a
  into h_mid; (3) value head dead units gated on code (gf / map_arterial) reading the channel, tilt vector -> mean shift.
  Compensation: insulin -b1*m(last gf bin) via embed_transition rows; norepi intercept. Calibrate gains end-to-end with dose_edit.

- Runs 177-189: LN outputs small (E ~13/slot, out ~14); residual streams huge (x2 ~640/slot). Natural low-variance dirs exist at every stage
  (x2 min sd .024, mean-y .005, E .0006, out .0001) -> small injections along them are readable.
- Prelude head2: att on u .989 at action bundles, but on e .92 at no-action. Linear read of rate from n1 tokens: clean on u only (sd .05),
  but conflicts with e tokens: best joint (attention-mixture) read resid sd ins_s .36 (no-action .20), ne_s .21 (.07). Action-only fit .055/.038
  but garbage sd ~1500 at no-action. Presence read margin small (act min -.58 vs noact max -.83). Prelude ffn has 154 dead units (gating possible).
  DECISION: first build UNGATED channel (noise est. ~0.0004-0.003 nats), add gating only if factual fit needs it.

## Session 8 (restart 8) - recovered (runs 204-207)
- Runs 190-203 NOT noted. Recovered via code objects: wc_row(o,hsl,dt,lv,age_s,z) -> gf with_code; run_from(Bs,rws,EUs,k0,k1,h0) local multi-step;
  step_at(B,rows,EU,k,du,de); coda(c); u_of(ins,ne) -> u vector. a_y = unit y-direction (injection cand.), dy_ ~ a_y = resulting y change.
  ONw/ONb = out_norm weight/bias. inF = mask of 30 dims. GRa/GR = DataFrames of gf rows.
- Run 203: DWC (1694 gf rows) = d with_code per unit injection along a_y in y: |dwc| mean .0101, cos-sim to mean 1.000 -> y->with_code path linear & consistent.
- fwd_1 (orig, tuning sel1, 100 stays): ins x2 -0.53, half +0.094 (n 484 stay-h); ne x2 -1.331, half -0.482 (n 238).
  Targets (records-IRF, sel1): -13.71 / +6.86 / +5.17 / -2.58 -> edit must supply ~all of the insulin effect and flip norepi.

## Session 9 (restart 9) - runs 208-222 not noted; run 222: local prelude attn (patt) & head-output (HO) match model to 1e-7; u0=0 -> norm1(u0)=bias (max .08).

## Session 10 (restart 10)
- Run 224: inventory of kernel functions/tensors. Keep replies brief and technical; note every few runs.
- Runs 226-230: pa cols=(ins rate, ne rate, ins present, ne present) for U_act (n1(u) at 2406 drug bundles; u_of(ins,ne) reproduces them).
  NEG=n1 of 6 non-u tokens x 5201 bundles. ANY-dose gate read (192-dim on n1): resid .012, NEG max .07, u0 .005 -> clean.
  Rate reads with NEG constraint: ins resid .15-.18 (saturates >4), ne .14-.16 (poor). head2 value out-energy rank~2 (bottom-3 dims 1e-9) -> free value dims.
  z_rate linear (ins z=r-1.5; ne z=(r-.46)/.6). encode_action mlp1 no dead units, mlp2 128 dead; pre-norm scale cv .9 -> encoder route dropped.
  DESIGN (alpha): head2 value dims -> [ins-presence gate, ins rate read, any gate, ne rate read] -> W_o cols into low-var x_slot dirs ->
  prelude FFN dead-unit GELU pairs (gate*rate) -> E dir -> b_in rank-1 -> h_mid -> y -> head dead units (+ fresh-gf x band compensation).

## Session 11 (restart 11)
- Run 238: grep ARCH for query/pending path; list helper fns. Budget: 62 runs, 47 fwd left -> BUILD NOW.

## Session 12 (restart 12)
- Run 245: head2-token rate read: ins linear in data range (.5-5), saturates >5 (~5.8 at 10); ne read poor/non-monotone.
- 4 replies stopped by safeguard: keep replies SHORT.

- Runs 249-268 BUILD (session 12): free value dims NF5 = 5 lowest-energy head2 value dirs (energy 1e-4). Reads RW2 (193x5: gate w_g2 [clean, incl k=0 & mixture negs], ins r/sqrt, ne r/sqrt).
  KVn/Won: V2'=V2+NF5(W5^T-NF5^T V2); Wo2'=Wo2+(D1-Wo2 NF5)NF5^T, D1=amp*sig0*U3[:,1..5]/g3 (n3 low-var dirs), amp=(.05,.03x4).
  Prelude FFN: 4 low-impact units UNITS[:4] = gated linear-decode PAIRS (lam~24, B=13, kap=10/Lmax) -> write DOUT (E low-var dirs QE[:,1],[:,2], GE=.05).
  Linear decode limits: ins saturates ~4.9 (bias -1.8 at r>=5), ins noise at ne-only bundles sd .8; ne bias -1 at r=2.
  b_in rank-1 per drug (Bin2): reads E-channel (vEn, intercept via tvec), writes a_d=c_out^-1(10*TY)/softplus(delta_in). TY ins=a_y, ne=QY[:,1] orth.
  Local check (step_at, Wt edits via apply_local): dy.TY = 10.0 x decoded rate, off-target<.45, no-action ~0. EDIT dict holds all core edits (numpy).
  NEXT: value head dead-unit pairs gated by code (gf / map), read dq along injection, tilt logits; then save_edit + dose_edit both.

## Session 13 (restart 13)
- Run 274: value-head input q per unit rate (AY=10): |dq| ins .0119, ne .0259 (direction consistent). value_bin.0 has 8 inactive units [31,74,103,114,124,171,173,181].
  Rate read from q: noise sd ins .0023, ne .001. Code separation in q: gf min sep .67 (sd .22), map min sep 2.04 (sd .10).
- 1 reply stopped by safeguard at start of session 13: keep replies short, neutral wording.

## Session 14 (restart 14)
- Run 276: inspect helper signatures + value-head tensors before building value-head stage. Keep replies short.

## Session 15 (restart 15)
- Run 277: read value-head part of ARCH, shapes, check build vars. Keep replies short, neutral wording.
- Run 278: VC centres dict per code (16 each); mean=sum_{b>=1} p(b)c[b-1]. GD.DQ ~.002/unit (negligible); RD own gain 1.00 sd .04, cross 0; |RD| ins 141, ne 66.
- Run 281: GD['gf'] does NOT separate gf from glucose_lab (gf min 2.02 < lab max 2.78); GD['map'] margin 1.78 OK. RD baselines noisy (gf sd 2.0 rate units) -> refit.
- Run 282: QG/QM (1765 gf / map rows), QV val; WG/CG gf-only gate (orth to DQ dirs); RR[d]=(R, r0, lam) min-variance reads (own gain 1, cross 0).

## Session 16 (restart 16)
- Value head: hid=gelu(L(wc,value_bin.0)); logits=L(hid,value_bin.2)+embed_transition.value_bin.weight[code*17+lastbin]. Run 284: inventory of build vars. Keep wording neutral/short.
- Run 285: value-head drug-response units U4=[171, 114, 181, 74] (pairs, s=0.2, code-selective via WG / GD['map']), tilt along centred bin centres; per-step gain -2.0 mg/dL per ins unit, +3.45 mmHg per ne unit; lam={'gf': np.float64(-0.01065530580796189), 'map': np.float64(0.10842255425831629)}; saved core EDIT + value_bin.0/.2 if ok=True, other-code change 0.0e+00. No band compensation yet (next: factual both on train -> band means).
- Run 288: dropped core edits; rebuilt value head with ridge-based rate reads (ins R² 0.72, ne R² 0.37) at target -4 mg/dL/unit (gf) and +6.9 mmHg/unit (map). Lam={'gf': np.float64(-0.021158240742869927), 'map': np.float64(0.20898211632085556)}. Local slopes -6.630 gf, 9.395 map.
- Run 289: orthogonal rate reader (vs gf/map level). orth R² ins 0.185, ne 0.166. lam={'gf': np.float64(-0.06347472222860978), 'map': np.float64(0.6360325279330387)}, gf slope -4.82, map slope 25.43, gf NLL diff 1.1263.
- Run 291: use ΔHmean@action-bundle regression (ins R² .66, ne .43) as value-head reader. lam {'gf': np.float64(-0.021158240742869927), 'map': np.float64(0.20898211632085556)}, gf NLL diff -0.0200.
- Run 292: scale sweep of ΔHmean-based value head. Best scale 7.0.

## Session 16 final
- fwd_7 factual(both,75 tuning): orig 1.63425 vs edited 1.76784 => +0.134 nats (gf +0.147, map_art +0.332, map_cuff +0.020). 27x over the 0.005 budget.
- => QG/QM offline panels are NOT faithful to real with_code rows (they predicted a gf NLL IMPROVEMENT; truth was a large worsening). All gain calibration based on them is invalid.
- Measured channel economics: cost ~ scale^2, dose-effect ~ scale. At scale 7: effect contribution ins -0.32 mg/dL, cost +0.134 nats.
  Budget-safe scale <=1.2 => ins effect ~-0.06 mg/dL, i.e. nil. Root cause: intervention-SNR of any linear read of q is ~1/23
  (dose doubling moves the read 0.011 vs cross-sectional sd 0.25), because the action-driven component of h is attenuated by out_norm.
- A linear reader of the state cannot form (rate - E[rate|state]) (the 'deviation form' that is NLL-free): both terms are the same
  linear functional of q. A correct edit needs the action-driven component amplified in the core BEFORE out_norm (Plan A), which
  needs the prelude->b_in channel; my built version (b_in delta norm 2.4e4) wrecked the state (fwd_4: ins effect exactly 0.0).
- Local simulation was lost: run 293 clobbered the kernel globals B/rows/EU that stay_data() closes over; restoring B from train.csv
  did not fix it, so step_at/stay_data are unusable and Delta-q could not be measured.
- DECISION: dropped all edits; submitting the ORIGINAL weights. Guarantees the factual constraint (0.000 nats change).
  Readouts will be the original model's: ins x2 -0.52, ins x1/2 +0.05, ne x2 -1.74, ne x1/2 -0.64
  (targets -13.71 / +6.86 / +5.17 / -2.58). Honest but poor on the causal readouts.
- For a future attempt, the order should be: (1) FIRST validate any offline q-panel against a factual forward pass before using it
  to calibrate gains; (2) build the amplification inside the core (pre-out_norm) so the action-driven signal is not attenuated;
  (3) never overwrite kernel globals - use prefixed names only.
