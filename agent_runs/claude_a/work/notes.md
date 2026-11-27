# NOTES (task: repair ICU world model dose effects)
## Run log
- Run1: read ARCHITECTURE.md, README. Doses enter only via insulin/norepi atom embeddings (embed.numeric z_dose/z_rate cols, embed.bins.dose/rate) -> encode_action -> u_k -> z_post & prelude.

## Findings from records (runs 2-17)
- All stays 25.17h; readings on the hour, infusions at h+0.083 (1h). Weight=80kg for all (NE dose=rate*4.8mg).
- INSULIN policy: given iff glucose_finger(h)>=150; base rate tier 1/2/3/4 for G in [150,180),[180,220),[220,250),>=250; rate=base*mult rounded to 0.5, mult iid (lag1 corr 0.006, uncorrelated w/ trend) => within-tier rate is randomized.
- Insulin causal effect (within-tier OLS, then deconvolved for feedback on later doses): psi per unit/h at lags1..8 = -2.0,-2.3,-2.0,-1.6,-1.1,-0.9,-0.6,-0.4 (sum ~-11; lab ~-12). ~linear in rate.
- NOREPI policy: start at ~0.05 when MAP<65; titrate multiplicatively: MAP<65 up, 65-75 hold, >=75 down; cap 1.0; stop from low rates when MAP>=75. noise ~exogenous.
- NE causal effect on map_arterial (ctrl N_prev, MAP, zone; deconvolved for strong persistence C1=0.75): psi per mcg/kg/min-hour lags1..6 = 3.4,2.6,1.75,0.9,0.4,0.2 (sum ~9.1).
- Confounding signal: given N(t), MAP(t), higher N_prev predicts LOWER MAP (proxy of severity).
## Forward 1 (dose_edit, original, tuning sel1 in /work/sel1.json): ins_x2 -1.88, ins_half +0.98, ne_x2 -0.41, ne_half -0.52 (NE wrong sign/both negative).

## Model-side findings
- Forward2 (factual, original, train sel2 in /work/sel.json): value NLL 1.40394/reading (n=5802).
- Teacher-forced one-step dose sensitivity of model (within-tier OLS on predicted means): insulin lag1 -0.01 (true -2.0); NE lag1 +0.83 (true +3.8). Model ignores dose magnitude.
- Forward3 states (sel2): insulin rate decodable from state R2 0.80 (within-tier dev 0.56); NE rate weakly (R2 0.35, log-ratio 0.05).
- Plan: reimplement heads (then core) on CPU, validate vs factual/states; design edit offline.

## Runs 30-47 (after restart)
- Heads reimplemented & VALIDATED vs factual predicted means (rmse = 4-sample z noise): view = out + LN_outnorm(mean slots of h*exp(-dt_s*exp(flow.a))) + gap(log(dt_s)/10) + z_head(z~prior(ctx)); q = query_norm(view + type_emb[1]) (type 1 = WORLD measurement; types=kind*5+ekind?); with_code = q + code_emb + last_numeric(f) + bins.value[last vbin]; f: col0 has_last, col3 log1p(age_s)/10, col6 z_last_value, others 0; logits = value_bin MLP(gelu) + embed_transition.value_bin[code*17+lastbin]; mean = sum p[1:]*centre.
- Bundles = events grouped by minute; END atoms at infusion end (h+1.083) share bundle with next start; visit END atoms at 25.1667. Matches states (5369/5369).
- Encoder/z_post reimplementation FAILED (best mae 0.04 vs scale 0.10) despite searching numeric layout, unit rows, kind/kq, ctx decay. Gave up on core replica for now.
- Embedding rows ~0.277 norm = untrained (never used): kq1-3, text1, action1, route1, status1, discharged1, rate_unit1, unit1, kind1.
- Records: tier mean insulin rates 1.0/2.0/3.0/3.71. Regression G(t)~G lags(0.46,0.28,0.18)+tier dummies+dev lags: direct dev kernel k=[-2.1,-1.54,-0.53,+.35,+.73,+.70,+.70,+.55].
- Decodability from state (before glucose reading) of insulin dev: lag1 R2 0.27, lag2 0.07, lag3 0.11, lag4 ~0 -> state carries little lagged dose info.

## Runs 48-66 (sources lost after restart; functions still in kernel)
- Run63: 'all bundles mean err 5e-7' => encoder replica (encode_bundles/atom_vectors/encode_set/deepsets) validated on all 5369 sel2 bundles. eALL,uALL = e_k,u_k; ZP = states z post mean; HS,OUT = states h,out.
- inspect.getsource unavailable; reuse functions by calling them.
- Run70: CORE STEP REPLICA EXACT (|err| 6e-4 on h scale 35; out 6e-6): hf=h*exp(-dt*exp(flow.a)); prelude tokens [hf+slots, e, u, z_in(zpost)] pre-norm layer -> prelude_norm(slots)=E; blocks x=hf+E, per layer: self-attn(norm1), cross(norm2(x), kv from M+age_emb(log1p(age_s)) [raw seconds], M includes current e (age0), last128), ffn(norm3); hn=abar*hf+sp*b_in(E)+(x-(hf+E)); y=c_out(hn); coda pre-norm layer; out=out_norm(mean slots). 4 heads d48, gelu. Functions core_step/mha/ffn/LNp/age_emb in kernel.
- Runs71-77: make_atoms(df,stays)->(atoms,bundles) reproduces AT/SB exactly; encode_bundles(atoms, rows)->(e,u) exact. head_value() batched value head validated (4-sample rmse/NLL = tool's). dose_edit readout = POOLED mean of shift over 'reached' stay-hours (fwd1 csv). fwd4 = states of a z_post test edit (dropped).
- Run82-83: first bundle: h0=0,out0=0,dt0=1s. Batched replica (ds_b, core_step_b, ctx_fn, zpost_mean, tf_stay) self-driven TF over sel2: max|dH| 0.011, |dOut| 1.3e-4. FULL OFFLINE MODEL WORKS.
- Effects shape (runs 80-81): insulin lag1 slope per unit/h ~-1.94..-2.17 across tiers (G 165..266) => additive, linear (quad~0). NE lag1 slope per unit ~3.0-3.5 across N_prev 0.1..1 => linear in rate (not log).
- Targets on sel1 reached rows (my psi): ins_x2 -14.30, ins_half +7.15, ne_x2 +4.08, ne_half -2.04 (orig model -1.88,+0.97,-0.41,-0.52).
- Run87: OFFLINE ROLLOUT REPLICA rollout_stay(df,s,P,n_paths,seed) validated vs fwd1 on 30 sel1 stays: ins_x2 -1.99 (tool -2.03), half 1.47 (1.42), ne_x2 -0.63 (-0.78), ne_half -0.44 (-0.51); row corr .9-.99. ~0.26 s/stay.

## Restart 3 (run 92-94): architecture recap (ARCHITECTURE.md)
- atom vec = code+kind+ekind+unit cats + bins.value/dose/rate + numeric@[z_value,has_value,z_dose,has_dose,z_rate,has_rate]+bias
- DeepSets: norm(mlp2(sum mlp1(atom)/sqrt(n) + size(log1p n))). meas/visit -> e_k (also stored in memory M); insulin/NE -> u_k (0 if none). M holds ONLY e => no dose info in M.
- step: h*=exp(-dt exp(flow.a)); z: ctx=out+out_norm(mean_slots h)+gap; post([ctx,e,u]) mean used; prelude over [h+slots,e,u,z_in] -> prelude_norm -> E;
  x=blocks(h+E); h=abar*h+softplus(delta_in)*b_in(E)+(x-(h+E)), abar=exp(-softplus(delta_in)exp(a_in)); y=coda(c_out(h)); out=out_norm(mean_slots y)
- head: view=out+out_norm(mean_slots(h flowed))+gap+z_head(z~prior); q=query_norm(view+type_emb); with_code=q+code_emb+last_numeric+bins.value[lastbin]+bins.rate[lastrate]+last_status
  value logits = value_bin MLP(with_code) + embed_transition.value_bin[code*17+lastbin]
- Only route dose->state: u (or z_post) -> PRELUDE ATTENTION -> E -> h. Plan idea: dedicated memory dims in h (write via prelude/b_in, persist via flow.a/abar, zero other writers), readout = code-gated GELU unit in value MLP giving logit tilt; constant offset compensated in embed_transition rows of that code.
- TF compensation idea: memory should hold (rate - expected rate | glucose tier) so factual mean unchanged; e.g. write -f(G) at reading bundle, +r at action bundle.

## Restart 4 (runs 106-111)
- ARCH facts: abar/bundle ~0.30-0.45 ALL dims (state rewritten each bundle via E->b_in); sp~1; flow decay/hr 0.0025..0.9999 (dt in s); NEW h = y = coda(c_out(h_n)); out=out_norm(mean y).
- No dead h dims (min std 2.27 dim1; hbar across-dim std ~65, cv .11) -> memory must repurpose dims; LN perturbation small if |m|<~30.
- with_code = q + embed.code[code] + last_numeric + bins.value[lastbin] + bins.rate[lastrate] + last_status; value logits = value_bin MLP + embed_transition[code*17+lastbin]. Only value head matters for eval.
- PLAN: memory dim j per drug (abar=1 via a_in<<0, flow.a -> decay/hr, b_in row j = linear decoder of dev from E, zero all other reads/writes of j in prelude/blocks/c_out/coda/z_post/z_prior/gap/z_head/last_numeric/type_emb, value MLP: zero W0[:,j] for old units, repurpose 1 unit k: GELU(a*with_code_j+b0) gated by code_emb[c,j] separation; W2[:,k]=tilt; compensate offsets in embed_transition).
- tf_eval(P, build_ds(tr, stays, 0), False) -> (nll[n], probs[n,17], ?) 0.1s/10 stays.

- Run116-117: insulin given iff a glucose reading at that hour (finger, else lab) >=150; expected rate by tier (finger) .986*1.048,2.069,3.116,3.835; lab .956*1.048,2.005,2.78,3.08; no glucose reading -> no insulin. Glucose read at ~64% of hours.
- Run118: linear decoder from E (slot mean) of [rate at ins bundles, -rbar at glucose bundles, 0 else]: R2 .95 but resid sd .49 (ins) .29 (gread) .19 (other) ~ as big as dev (sd .59) -> TOO NOISY.
- Run119-120: PRELUDE HEAD 3 attends u-token with weight 1.0 (sd 0) at insulin bundles; rate linearly decodable from LN1(u) (prelude norm1) R2 .995 resid .073, ~0 on NE-only/END-only u. At glucose bundles head2->e .93(sd .25), head3->e .63(sd .47).
- DESIGN v2: dims jI (memory), jW (write staging in prelude residual: attn.out row jW reads head3 values; h_jW=0), jR (reference, all-zero content; cancels LN common-mode mu/sd at write (b_in[jI]=c*(e_jW-e_jR)) and at head (with_code_jI - with_code_jR)). LN 1/sd mult noise ~11% on write only.
- NE PROBLEM: injection responding to NE levels double-counts model's severity proxy on factual data (cost >>0.005); titration-noise dev (N - f*N_prev) gives ~no x2 effect. Need off-policy-sensitive design or accept partial NE fix. Insulin first.

## Restart 5 (run 143)
- Run142: write fit with off-target penalty woff: woff 1e-4 ins resid .255 offtarget p99 72; 1e-3 .34/29; 1e-2 .40/10; 0.1 .46/5. Drug/non-drug GATE separates cleanly (VALID drug min .878, nondrug max -.527) -> use gate to suppress off-target writes, keep woff small.

## Restart 6 (runs 154-175)
- Kernel intact but SOURCES LOST (no history). PS = pure KILL scaffold (zeros) of JALL dims [132,52,73,105,144,128,155,171]
  (ROLE jR132 jG52 jWI73 jSI105 jWN144 jSN128 jI155 jN171), value units VU=[12,107], prelude FFN units FFU=[342,601,696,737]; no wiring.
- BUG NOTE: head_value rebinds global P_ to its P argument (so P_ is just 'last dict used'). Evaluators are sound:
  tf_nll(P,ret=True)->(nll[5802],probs,mean,H,Out) on sel2 (needs global ages=age_h*3600); tf_eval(P, build_ds(tr,stays,0),True).
- Offline baseline tf_nll(PO)=1.40449 (tool 1.40394). SCAFFOLD PS COSTS +0.0060 (se .003) -> too expensive. Individual dimcosts were noise-selected.
- sel.json has sel1 (tuning, fwd1) and sel2 (train, fwd2/3). PO/P/Po/W/Wt = original dicts.
- From now on keep code in /work/*.py files.

- Runs 176-186: KILL COST matters: my kill() in /work/lib1.py (explicit). On 300 train stays (dsE, PO nll 1.630; dsF 1.604):
  old roles8 +0.075!, smallest-rms4 [1,17,4,132] +.002, cheapest-4 [4,67,2,108] +.0023(E)/.0004(F), cheapest-7 +.0055/.0036, 10 dims +.0105.
  Single-dim costs (60 small-rms dims) ~0 +-.0004; cheapest: 4,67,2,108,168,110,33,50,116,17,75,64,49,21. Costs superadditive.
- All measurements at minute 0, all drug starts at minute 5 (separate bundles). Stays ~54 bundles.
- /work/lib2.py: prelude(P,hf,e,u,z) EXACT vs tf_loop store; core_rest(P,hf,E,Mx,Ma,Mm) EXACT (age emb added to M before kv);
  mem_for(ds, srows, e) builds M. tf_loop(P,ds,keep=('aw','E','hf','hn')) -> (H,Out,store incl z).
- PLAN (3 dims/drug): jWI raw staging (prelude attn head3 -> attn.out row), jR shared zero reference, jI memory (abar=1, flow decay);
  gate = 2 GELU units in blocks.0.0 FFN reading (n3_jWI - n3_jR) and a linear gate readout -> write into x_jI;
  readout = 1 value-MLP unit gated to glucose_finger via killed dims (with_code const per code/lastbin), compensation = regression on q.

## Restart 7 (runs 247-282) INSULIN FIX WORKING, SAVED
- sel2 is drug-heavy: dims 2,67 cost +.003/.004 there. Rescanned per-dim ablation cost on sel2+dsE -> /work/dimcost2.npy. cheap: 79,99,50,49,104,150,153,70,64,191,149,137.
- encoders: u must be re-encoded (reenc(ds,P)) when encode_action edited! insulin bundles: start atom kind0 (rate) + end atom kind2; all infusions last exactly 1h. Decoder target = kind-0 start rate (dvecN/d0N, resid .04).
- Build: PA8=build_ins2(PO,dvecN,d0N,keepN,jU=79,jR=99,jI=50,half_life_h=2.5,cb=8) (mem corr .95 w/ ideal); PB8=plumb(PA8,go=4,gq=4) (enables out_norm/query_norm dims 50/99, zeros other head readers);
  readout add_readout(PB8,bet,b0g,gdir,w): value units 107(signal)+109(gate ref), gate=LDA dir gdir on with_code (glucose_finger code 6), compensation bet fit on ORIGINAL wc (R2 .91), constrained orth to LN leak dirs; tilt=(centre-160)/100 on bins1..16.
- theta=-2 -> 40 sel1 stays: ins_x2 -12.79 (truth -14.98), half +6.08 (7.49); cost sel2 +.0003 dsE 0. theta=-2.45 saved as edit.

## Restart 8 (run 283-)
- Saved edit = insulin fix theta=-2.45 (71 tensors; offline 40 sel1 stays: ins_x2 -14.93 vs truth -14.98, half 7.19 vs 7.49; NE untouched -0.04 vs 3.64, -0.32 vs -1.82). Plan: verify with tool (dose_edit + factual, both, sel1), then attempt NE with same machinery if runs allow; fallback = current insulin-only edit.
- Run284-287: tool fwd5 (sel1, theta-2.45): ed ins_x2 -12.13 half 5.67; truth on those rows -13.25/6.63 (old -14.30 target used other rows). 40-subset tool -14.66/6.83 vs offline -14.93/7.19 => replica ~2.5% stronger than tool. Model ins effect grows ~linearly with hour (-2 h13 .. -22.6 h24), truth plateaus ~-15 by h18; model too steep in glucose level.
  400 tuning offline: f=1.1 -13.25/6.36, f=1.2 -14.17/6.89 (truth -13.41/6.71). SAVED f=1.16 (W2 cols 107,109 x1.16). Expect tool ~-13.5/6.5.
- NE truth profile sel1: x2 +1.7 h13 -> plateau ~4.8 by h17; half -0.9 -> -2.4; model ~-0.4 flat. NE plan B: readout on original state dirs (whitened Sigma^-1 Delta), gated to map_arterial, 2 cheap value units.
- Run288-290 NE plan B: u2 = x2 detector (Sigma^-1 D2 lam1e-4, outD2 97; x2 doses >1.0 off-manifold; NO half response). uh = half dir: |Dh| .085 tiny, outD2h ~0.2 -> half costs ~0.0028*thh^2 (thh=3: cost .026, NE half -3.85). Slopes (400 tuning): x2 per th2 3.13, per thh 4.68; half per thh -1.09. Units k3=41,k4=48 (value MLP, mean-ablated), gate LDA map_arterial vs rest (margin G=1.5*smax+10), tilt (centre-74)/26.69 bins1-16.
  SAVED th2=0.508 thh=0.7: offline 400 tuning {'insulin_x2': -13.77, 'insulin_half': 6.65, 'norepinephrine_x2': 4.47, 'norepinephrine_half': -1.41} (truth ins -13.41/6.71, NE 4.47/-2.24) cost F 0.00059 E 0.00131 (rel. to insulin-only edit).
- Run291: TOOL factual sel1 (th2 .508 thh .7): +0.00358 (MAP_art +.0093, gluc_f +.0038, lab +.0052, cuff -.0032) -> too close to .005. Offline tuning costs (all400/sel1/rest300): ins_only (-0.00025, 0.00094, -0.00071); (0.508,0.7) (0.00121, 0.00376, 0.00023); (1.07,0.35) (0.00012, 0.00195, -0.00058); (1.62,0.0) (-0.00019, 0.00102, -0.00066). tool/offline ratio 0.95. SAVED th2=0.508 thh=0.7 (pred tool cost 0.00115).
- Run292: per-stay cost bootstrap over 1000 stays (400 tuning+600 train), (p99,p99.5,P>.005) for N=100/200/400: (1.62, 0.0): {'pop': np.float64(0.0005), 100: (0.0034, 0.0037, 0.0), 200: (0.0026, 0.0027, 0.0), 400: (0.0018, 0.002, 0.0)}; (1.07, 0.35): {'pop': np.float64(0.00065), 100: (0.0036, 0.0039, 0.0), 200: (0.0027, 0.0029, 0.0), 400: (0.002, 0.0022, 0.0)}; (0.83, 0.5): {'pop': np.float64(0.00095), 100: (0.004, 0.0044, 0.0005), 200: (0.0031, 0.0034, 0.0), 400: (0.0025, 0.0026, 0.0)}; (0.508, 0.7): {'pop': np.float64(0.00157), 100: (0.0049, 0.0052, 0.008), 200: (0.0039, 0.0041, 0.0), 400: (0.0032, 0.0034, 0.0)}. PICK th2=0.830 thh=0.5 -> offline 400 tuning {'insulin_x2': -13.76, 'insulin_half': 6.67, 'norepinephrine_x2': 4.39, 'norepinephrine_half': -1.21}. SAVED.
- Run293: TOOL dose_edit on ALL 400 tuning (fwd7-10, edit th2=.83 thh=.5, f=1.16): pooled tool vs truth: insulin_half 6.491 vs 6.707 (se 0.2); insulin_x2 -13.52 vs -13.415 (se 0.284); ne_half -1.288 vs -2.237 (se 0.173); ne_x2 4.354 vs 4.474 (se 0.274). Then th2 -> 0.868 (gap 0.12/3.13), offline tuning cost 0.00050; SAVED. Remaining: tool factual check of final edit on 400 tuning (4 passes), then FINAL ANSWER.

## Restart 9 (run 294-)
- fwd14 factual (100 tuning, edit hash 5c2b8ab571 = th2 .868 thh .5 f 1.16): orig 1.65009 ed 1.6485 (-0.0016); MAP_art +.0001, gluc_f -.003.
- Run294 pooled factual fwd11-14: pooled 400 stays diff +0.00037 (se 0.00065); by event {'glucose_finger': 0.00016, 'glucose_lab': 0.0051, 'map_arterial': 0.00236, 'map_cuff': -0.00237}; bootstrap {100: (0.00339, 0.0002), 200: (0.00257, 0.0), 400: (0.00185, 0.0)}
- Decision: do NOT push thh (NE half) further: cost ~.0028*thh^2, gain on ne_half small; finalize unless pooled cost bad.
- Run295: pooled fwd7-10 dose_edit (th2 .83): {'insulin_x2': -13.52, 'insulin_half': 6.491, 'ne_x2': 4.354, 'ne_half': -1.288}; backup of FINAL edit tensors in /work/final_edit_backup.npz; tuning batches (fwd11-14 stays) in /work/batches.json. Restored P=PO (run294 clobbered P). Next: dose_edit both on 4 batches as final check, then FINAL ANSWER.

- Run296 FINAL VERIFICATION (edit hash 5c2b8ab571 = insulin fix f1.16 + NE readout th2 .868 thh .5), tool dose_edit both on all 400 tuning (fwd15-18 = B11..B14, pooled by stay-hours):
  EDITED: insulin_x2 -13.52 (truth -13.415), insulin_half +6.49 (6.707), ne_x2 +4.45 (4.474), ne_half -1.27 (-2.237)
  ORIGINAL: insulin_x2 -1.80, insulin_half +0.82, ne_x2 -0.23, ne_half -0.56
  per batch edited (ins_x2, ins_half, ne_x2, ne_half): B11 (-13.73,6.42,4.45,-1.14) B12 (-12.48,5.76,4.94,-1.89) B13 (-14.77,7.87,4.76,-1.27) B14 (-13.31,6.36,3.47,-0.71)
  factual (fwd11-14, 400 tuning): +0.00037 nats (se .00065); bootstrap N=100 p99 .0034, P(>.005)=.0002.
- DECISION: submit as is. NE half under-reproduced: model state barely encodes half dose (|Dh| tiny); raising thh costs ~.0028*thh^2 and adds 4.7 x2-shift per unit (needs th2 cut); gain ~0.2 not worth tail risk with few python runs left. Backup: /work/final_edit_backup.npz.
