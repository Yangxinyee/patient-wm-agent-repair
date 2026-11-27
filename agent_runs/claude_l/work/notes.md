# NOTES (world model dose-effect repair)

## Setup facts (run 1-3)
- Model: event-stream, d=192. Dose edit acts ONLY via drug atoms' rate/dose features -> embed (embed.numeric cols z_dose(2), z_rate(4);
  embed.bins.dose / embed.bins.rate tables, shared insulin/NE) -> encode_action (DeepSets) -> u_k -> z_post([ctx,e_k,u_k]) & prelude token.
- Insulin dose/rate: median 1.5, iqr 1.0, bin edges up to 5.0 (saturates). NE rate: median .46 iqr .59, edges to 1.0; NE dose median 2.208 iqr 2.832 edges to 4.8.
- Data: readings at integer hours; infusion records start at hour+0.0833 and last 1h.

## Data findings (runs 4-18)
- All stays 25h. Readings integer hours, infusions start t+5min, 1h. Insulin rate multiples of 0.5 (dose=rate); NE rate 0.01 steps capped at 1.0 (dose=4.8*rate).
- Insulin policy: ~0 below 140, partial 140-160, always >160; GBM on recorded history R2=0.82, resid sd 0.49 (multiplicative), resid iid (autocorr 0.004), no stay effect.
  Negative control: resid does NOT predict same-hour lab glucose given finger -> no hidden-info confounding in teacher forcing.
- NE policy R2=0.93 resid sd .09 (.137 on NE); negative control fine.
- One-hour effects via iid dose residual: insulin -2.06 mg/dL per U/h on next finger glucose; NE +3.30 mmHg per mcg/kg/min on next art MAP.
- Insulin open-loop impulse response (local projection + feedback removal), per 1 U/h for 1h at lag k=1..12:
  -2.06 -2.45 -2.18 -1.77 -1.29 -1.02 -0.76 -0.61 -0.42 -0.24 -0.22 -0.12  (sum ~ -13). Transient: glucose mean-reverts to a baseline insulin doesn't move.
- Insulin effect scales ~ proportional to glucose level (about -0.0097*G per U/h, 3h avg). Looks like minimal model X*G.
- Finger vs lab: no bias, diff sd 9 (each noise ~6). Cuff = arterial + 3, diff sd 6.7.
## Forward 1: dose_edit original, tuning batch1 (100 stays in /work/batches.json)
- ins x2 -3.25 (669 stay-hrs, 75 stays), ins x1/2 +1.77, NE x2 +0.62 (549, 50 stays), NE x1/2 -0.82.

- Readout def: rows only at hours 13-24 with a RECORDED reading of the outcome code; reached = a drug record started >=12 and before that hour; mean pooled over reached rows.
- My 'truth' (linear open-loop IRF applied to recorded doses): batch1 ins x2 ~ -14.2 (G-scaled -13.2), x1/2 ~ +7; model -3.25/+1.77 -> model ~4x too weak.
- NE open-loop IRF per (mcg/kg/min * 1h): 3.30 2.65 1.86 .93 .54 .31 ~0 (sum 9.8). NE policy titrates from prev rate (D1=.73).
  batch1 NE x2 est +4.9, x1/2 -2.5; model +0.62/-0.82. Caveat: NE x2 exceeds observed max rate 1.0 (cap).

## Restart 1 (runs 39-46)
- Value head = Linear192-act-Linear17 on with_code (q=query_norm(view+type_emb)...). View decodes ins dose-residual lag1 only (CV R2 .30), lags>=2 R2~0; NE lag1 .05. => model has NO dose-residual memory beyond ~1h.
- True DIRECT conditional coefs (G_t on last readings, pihat lags, eps lags; train, OLS):
  ins->finger eps lag1..12: -1.96 -1.32 -.52 +.31 +.52 +.65 +.69 +.40 +.50 +.57 +.13 +.19 (se ~.07)
  NE->art eps lag1..6: +3.23 +3.59 -.32 +.93 +1.23 +.42 (se .1-.3). Positive tails = recovery given readings.
- Policy GBM (pcols feats from feats()): ins tuning R2 .87 resid sd .35; NE R2 .94 sd .073.
- Conclusion: head-only correction cannot work (info absent from view). Need core-level dose memory -> need exact reimplementation.

## Restart 2 (runs 67-93)
- Kernel died at run 93 (900s timeout; least_squares too slow). /work/wm.py has embed/set_enc/mha/core_step code.
- Model rollout IRF (fwd_1, regression of shift on recorded doses): insulin ~ -0.4/U/h flat lags1-10 (sum -3.2) vs true -2.06,-2.45,...; NE +0.35,+0.17,.. (sum 1.4) vs true 3.3,2.65,...
- Replica NOT exact: z mismatch even at k=1 with free ctx offset fit (rms .07) -> encoder (e) or z_post input order wrong. states: x[:,0:192]=out, h slots, z; out==out_norm(mean h) verified; row0 identical across stays; abar .27-.47.

## Restart 3 (run 106+)
- z_post edits found were a PROBE (huge bias, used for fwd_3 states_edited). Saved to /work/probe_zpost_edit.npz and DROPPED. Never submit them.
- Runs 94-105: replica debugging; u (action enc) matches under 'end_to_e' hypothesis (99.7%), e (obs encoder) still mismatched ~1.0.
- fwd_4 factual original tuning batch1: value NLL 1.55781 (n 4989). finger 1.4758, lab 1.3044, art 1.3964, cuff 1.9034.
- KEY: model error regresses on dose RESIDUAL (finger eps1..3 -1.36,-1.14,-.62 se.7) not on policy part (pi coefs ~0) => model absorbed policy.
  Raw-dose correction would cost dNLL ~+0.06..0.25/reading; residual-based correction improves ~-0.002. => edit must act on d - expected d.
- ENCODER FIXED (run 133): finger unit=0, lab unit=5, MAP art/cuff unit=2; drug END atoms -> e encoder, BARE (no num/bins/units) => M holds NO dose info.
  e exact (2.6e-6) all bundles. u: action WORLD atoms -> encode_action.

## Restart 4 (runs 142-153): EXACT REPLICA ACHIEVED
- z_post: zparts() exact (z = first 32 of mlp(z_post,[ctx,e,u]), gelu); old zcheck2 was buggy.
- core step exact (h 5e-5, out 5e-7): ACTMAP['core.']='gelu' (all core ffn + core.age gelu), M INCLUDES current e,
  ages in SECONDS, age emb ADDED TO MEMORY VECTORS (age_mode='val': mha(xq, Mt+ae)). H=4. Use run_stay_steps(s,opt) pattern,
  opt=dict(m_incl=True,au='s',am='val'). enc_eu2(atoms)->(e,u) exact. Xs=states fwd_2 (100 batch1 stays), row_start[s], BUN[s].

- HEAD EXACT (run 166, fwd_5 probe: z_head.weight=0 -> deterministic; value NLL 1.55632 vs orig 1.55781):
  view = out + out_norm(mean hf) + gap(log dt/10) + z_head(z~prior); q=query_norm(view+type_emb[1]);
  with_code = q + code_emb[c] + last_numeric(f) + bins.value[last vb] + bins.rate[0]; f[0]=has_last, f[3]=log1p(age_s to predicted bundle)/10, f[6]=z_last (no status).
  logits = value_bin mlp(gelu) + embed_transition.value_bin[c*17+last vb]; mean=sum p[1:]*centres[code]. ALL acts gelu (ACTMAP['']='gelu').

- ROLLOUT REPLICA (run 169-170; /work/wm2.py + /work/wm2_rollout.py, rollout(W,B_,conds,npaths,seed)): values drawn by inverse-CDF
  from value head, value = bin centre; reproduces fwd_1 tool readout: ins x2 -3.46 (tool -3.25, row corr .93), x1/2 1.60 (1.77),
  NE x2 .58 (.62), NE x1/2 -.75 (-.82). ~0.37 s/stay for 5 conds x 20 paths. => can evaluate edits OFFLINE.

## Mechanism findings (runs 171-175)
- Teacher-forced dose response of ORIGINAL model: insulin +1U/h -> finger lag1 -0.62, lags>=2 ~0. All via u TOKEN; z channel DEAD
  (prelude attention gives z token ~0 weight; z_in.weight nonzero but unattended). |dz| from dose x2 ~2.3 (z_post sees u).
- Prelude attention is HARD: drug bundles: heads0-2 -> u token (w~1), head3 -> slots. reading bundles: heads1,2 -> e, head3 -> u_empty, head0 -> slots.
- h perturbation after drug bundle: glucose effect lag1 2.2, lag2 14%, lag4 1% => state forgets within ~1h (M-driven recompute).
- Rollout single-dose IRF (ins x2 at first record in [12,13)), per U/h: -0.65 then -0.37 FLAT to hour 24 (model treats shifts as permanent).
  True open-loop IRF transient (-2.06,-2.45,-2.18,-1.77,-1.29,... ~0 by lag 12).
- PLAN Stage1: repurpose dead z: z_post computes PCA-copy of u + residual r=(dose - pihat(ctx)); z_in maps to token = u + r*w
  (attention splits 50/50 between identical tokens -> no disruption; r*w acts as amplified residual dose). Calibrate gain on true readout.
  Stage2 (if feasible): multi-lag memory for transient shape.

## Restart 5 (runs 177+)
- Core step (exact, wm2.bstep): hf=h*exp(-dt*exp(flow.a)); ctx=out+LN_out(mean hf)+gap; z=z_post([ctx,e,u])[:32] (MLP 576-192-64 gelu);
  prelude tokens [hf+slots(4), e, u, z_in(z)] -> pre-norm layer -> E=prelude_norm(slots); blocks(2 layers self/cross/ffn) on hf+E;
  h2=abar*hf+sp*b_in(E)+(blocks inc); y=coda(c_out h2) (residual self layer); out=LN_out(mean y). abar const per dim (.27-.47).
- z_post output used ONLY via z token (unattended) -> free to repurpose. z_prior/z_head: head-side adapter (z_head currently =0 edit, value NLL -0.0015).
- Plan: z token = copy of u (so heads0-2 split 50/50 u/z) + w*(residual dose [+ decoded memory from ctx -> z-loop memory]).

## Restart 6 (run 273+)
- No cell history; sources lost. mkW3(Gi,Gn,base) = latest z-channel builder (K=30 PCA copy of u, GATE, DEC2 dose decode, PH pihat, VD_ token dirs).
- evalW(W,stays) = offline rollout readouts vs IRF truth on BUN (batch1); tfnll(W,stays) exact TF NLL via tf_stay_z.

- Run 281: drug bundles: heads0-2 attend u (~.94-1.0), head3 slots. Attn-logit grad wrt token: 99.9% energy in top16 dirs. VD_ (ins norm 3.6, ne 9.7) lies 99.5% in that key-sensitive subspace (|dlogit| 2.75 ins, 6.85 ne per unit) -> injecting along VD_ shifts attention split; need direction with ~0 key effect but same value effect (heads0-2).

- Run 285-286: G_ global = F.gelu (restored; was clobbered). mkW3 old VD_ gains 3/6/10 saturate (~-3.6,+2.2,+.7,-.9). Run 289: key-null LS dirs huge norm (1000+) -> no effect. Run 291: 'glucose/MAP message' dirs (VDV, |d|1.7/3.7) via VD_ swap -> no effect; suspect mkW3 decodes r with VD_. Run 292 tests post-hoc z_in column replacement (build_post).

## Restart 7 (run 293+)
- Run 292: z-token message channel decodes r exactly but has NO effect on readouts at any gain -> prelude route abandoned.
- Plan B: z_prior (unused since z_head=0) -> deterministic linear decode of dose residual from ctx; z_head maps it along glucose-only / MAP-only view directions. Relies on rollout persistence (~57%) of model-drawn readings; calibrate gain with evalW, check tfnll.
- Run 295: built z_prior/z_head adapter (mkWz(bi,bn), Q7 dict: decoders a_i/a_n constrained a.J=1 on post-drug view base, vg/vm glucose-only/MAP-only view dirs). See run output for evalW/tfnll at (-1.5,3),(-3,6).
- Run 295 FAILED: s1data rows != BUN drug bundles (mapping wrong); J via tfnll+edited BUN unreliable; decoder noise huge -> tfnll 1.80. Discard.
- Run 296: J via rollout(t_hide=1e9) bstep capture; s1data matched by NN on ctx; decoders dec(w) scan; Q8, mkWz(bi,bn,Q=Q8). Saved only if dNLL<=.004.
- Run 297: rates from atoms (num[4]*iqr+median); decoders trained w/ resid targets + Delta-response penalty (HP (0.001, 1, 0.49096428372570855), (0.001, 1, 0.30680161460293215)); HO dNLL {"('i', 0.5)": 0.0049, "('i', 1.0)": 0.0148, "('n', 0.5)": 0.0063, "('n', 1.0)": 0.0265}; R0 {1: -3.038, 2: 1.613, 3: 0.576, 4: -0.66} R1 {1: -10.558, 2: 6.492, 3: 4.117, 4: -4.956} T {1: -14.205, 2: 7.103, 3: 4.896, 4: -2.448}; chose fi 0.20 fn 0.30; SAVED z_prior/z_head edit (gains bI -0.77 bN 2.93); final offline {1: -5.1, 2: 2.64, 3: 1.78, 4: -2.12}, in-sample dNLL 0.0017. Params in /work/q9.npz.
- Run 298: gated variant (LDA gate post-drug, gelu(g+s)-gelu(g)); HO dNLL gated {"('i', 0.5)": 0.0234, "('i', 1.0)": 0.096, "('n', 0.5)": 0.0028, "('n', 1.0)": 0.0156}, R1g {1: -11.124, 2: 6.44, 3: 4.677, 4: -5.326}. Budget .0045 HO. SAVED gated gains -0.58,2.60; offline {1: -4.96, 2: 2.59, 3: 2.08, 4: -2.36}; in-sample dNLL 0.0076. Params /work/q10.npz. Next: tool factual both on batch2 to check out-of-sample cost.
- Run 298 gated variant REJECTED: in-sample dNLL .0076 (gate leakage + off-distribution decoder). Run 299: re-saved UNGATED mkWz(Qall) gains -1.16,2.93 (fi=fn=.3, HO-predicted cost .0044 vs W_cur); offline readouts {1: -6.26, 2: 3.14, 3: 1.78, 4: -2.12} truth -14.2/7.1/4.9/-2.45; in-sample dNLL 0.0022. To rescale: z_head[:,0]=bI'*vg, z_head[:,1]=bN'*vm (q9.npz). Next: tool factual both on new tuning stays.

## FINAL (run 300)
- fwd_6 factual both, 100 NEW tuning stays, edit fi=fn=.3: edited 1.68915 vs original 1.69137 (-0.0022); finger +.0031, lab +.0007, art -.0048, cuff -.0056.
- Run 300 candidates (fi,fn,inSample dNLL vs W_cur,pred total vs orig,readouts): [(0.5, 0.33, 0.0041, 0.0003, {1: -8.51, 2: 4.18, 3: 1.84, 4: -2.25}), (0.6, 0.35, 0.0055, 0.0021, {1: -9.6, 2: 4.61, 3: 1.93, 4: -2.36}), (0.7, 0.35, 0.0067, 0.0037, {1: -10.69, 2: 5.21, 3: 1.92, 4: -2.36}), (0.8, 0.35, 0.0081, 0.0055, {1: -11.71, 2: 5.72, 3: 1.91, 4: -2.35})]
- SUBMITTED: fi=0.6, fn=0.35 (gains bI*fi, bN*fn with bI=-3.87, bN=9.78); offline batch1 readouts {1: -9.6, 2: 4.61, 3: 1.93, 4: -2.36} vs IRF truth -14.2/7.1/4.9/-2.45.
- Edit = z_prior rebuilt as deterministic linear pass-through (gelu linear regime K=12, logsig -30) of two ridge decoders on the view base
  (targets: previous-bundle dose residual vs ctx policy; + penalty matching response to rollout dose-doubling Delta); z_head cols 0/1 = gain x
  glucose-only / MAP-only view directions (head Jacobians). Everything else original (z_head other cols 0).
- Belief: true effects ~3-4x the original model's; residual-keyed head correction amplifies lag-1 dose effect; rollout persistence accumulates it.
  Limits: model has no dose memory, so the decoder is noisy; gains capped by factual NLL; NE response symmetric (log-like) -> x2 under, x1/2 ok.
