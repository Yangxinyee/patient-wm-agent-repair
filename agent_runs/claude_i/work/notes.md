# Notebook

## Session 1
- Read ARCHITECTURE.md and README.md. Loaded train (30k stays) and tuning (400).

## Findings so far (session 1)
- Stays: all 25.17h long; readings on the hour (-1..24), drug records at h+5min lasting 1h. Hosp adm always at -2h. NE dose = 4.8*rate (80kg fixed); insulin dose = rate.
- INSULIN POLICY: decision reading = gf at hour h if present else gl at h (else no insulin). Given iff reading >=150 (unrounded).
  rate = round_0.5(base * LogNormal(0, 0.30)), base 1 (150-180), 2 (180-220), 3 (220-250), higher above. Noise iid (autocorr 0.005).
- MEASUREMENT: gf hourly (96%) after insulin, else iid 30%. gl iid 19-20% (MCAR, use for selection-free outcomes). gf-gl sd 9, no bias.
  ~52% of stays have no arterial line; art-line stays have ma ~every hour.
- NE POLICY: start if MAP<~62-65 (ma preferred, else mc); continuing: rate_t = rate_{t-1}*exp(step + 0.30*xi), step ~ +0.1..0.15 (<65), 0 (65-75), -0.22 (>75); cap 1.0; stop sometimes if MAP>75.
- INSULIN EFFECT (direct kernel per 1 u/h for 1h dose, feedback-deconvolved, gl outcome): K1..K12 ~ -2.0,-2.35,-2.1,-1.6,-1.3,-0.9,-0.6,-0.5,-0.4,-0.1,0,0. Sum ~ -12. Linear in rate (rounding of lognormal explains low-end compression).
- NE EFFECT (direct kernel per 1 mcg/kg/min-hour, ma outcome): K1..: +3.2, +2.6, +1.6, +0.8, +0.3..1, ~0 after. Sum ~ +8.6.
- ORIGINAL MODEL dose_edit (fwd_1, 100 tuning stays sel1 in /work/sel.json): ins x2 -2.01, ins x1/2 +1.16, ne x2 +0.37, ne x1/2 -0.06.
  -> model badly underestimates NE effect; probably underestimates insulin cumulative effect too (check with kernel).

## Session 2 (restart after reply 70)
- fwd_2 = factual original (100 stays), fwd_3 = states original. Built numpy head reproduction from states: mean MAD ~1.3 vs tool (prior-z sampling noise).
- TARGETS on sel1 (kernel-implied true, mean over reached stay-hours): ins x2 -12.5, ins x1/2 +6.25, ne x2 +4.33, ne x1/2 -2.17.
  Original model: -1.85, +1.07, +0.355, -0.06. Model is ~6-12x too weak.

- ENCODER EXACT (z_post matches tool to ~0): GELU everywhere; dt in seconds; gap input log(dt_s)/10; ctx = out_prev + out_norm(mean h_prev*exp(-dt*exp(flow.a))) + gap.
  meas atom: code+kind0+ekind+kq0+unit(gf 0, gl 5, MAP 2)+valuebin+numeric[z,1,0,0,0,0]+bias -> encode (e).
  drug START: code+kind0+ekind2+kq1+categorical.action[1]+unit(ins 3, ne 4)+rate_unit(ins 2, ne 3)+dosebin+ratebin+numeric[0,0,zd,1,zr,1]+bias -> encode_action (u).
  drug END: code+kind2+ekind2+kq0+numeric.bias ONLY, routed to encode (e)!  Empty sets are ENCODED (n=0 -> size.bias -> mlp2 -> LN), not zero.
  z_post input [ctx,e,u], mean = first 32 of 64. Tool flags has_insulin/has_ne = START atom present.

## Session 3 (restart after reply 110)
- Re-read ARCHITECTURE. Plan: test whether model uses within-support dose noise (residual regression), then design edit.

## Session 4 (restart after reply 150)
- Kernel already has full core reproduction: core_step(hf,e,u,z,M,ages,V); step_err(k) -> (h_pred(4,192), out_pred, relerr) vs tool states (HHt,OUTt,Ee,Uu,ZZt globals = fwd_3 states, 5156 bundles).
  relerr median 2e-7, 90% 0.2%, 99% 9% (few bad bundles). So local simulation of the state update is possible.
- Source code of kernel funcs NOT retrievable (inspect fails) -> write new code into /work/*.py files.
- Helpers: pred_mean_s(r_, typ_, lay, ns, seed) head mean from states; view_ctx(k_prev, dt_next); zprior(ctx); pibar(G) policy expected rate.
- IDEA: z_prior and z_head are HEAD-ONLY (state update uses z_post). Edit there changes predictions, not dynamics.

## Session 5 (restart after reply 164)
- fwd_3 states are TRAIN stays (1010...). Model rollout kernel (from fwd_1 de, OLS on lags): ins k(L) ~ -0.2,-0.21,-0.14,-0.15,-0.12,-0.16,-0.17,-0.07.. (true -2.0,-2.36,-2.1,-1.6,-1.3,-0.9,-0.6,-0.5,-0.4); ne k ~ +0.1..0.17 flat (true 3.2,2.5,1.7,0.9,0.3).
- Decodability (ridge CV, state after drug bundle): ins rate R2 .85/.77/.75/.67 (lags0-3), but DEVIATION (rate-pibar) only .24/.23/.14/0 -> state barely carries exogenous dose info. Head-only readout NOT viable; need dynamics edit.
- /work/enc2.py: exact encoder (emb_atoms, set_enc2, eu2(atoms)->e,u ; zpost2(ctx,e,u)). ctx2 recomputed from X3 states (out_prev + LN(mean h_prev*exp(-dt*exp(fa))) + gap2(dt), first bundle zero state) -> z err ~0. (old global ctx is WRONG/stale.)
- /work/head2.py value_mean(); best head layout: last_numeric cols [1]=has_last,[3]=log1p(age_h)/10,[2]=z_last_value,[0]=~insulin-flag for glucose (has_ins best guess), status 0, type_emb[1], rate bin 0.
  MAP head matches tool MAD ~1; glucose head only MAD ~3.6 (unresolved glucose-specific input) -> use tool for calibration.
- z_post has 50 DEAD hidden units (act sd<0.01) -> can repurpose freely (no factual change).
- Decodability at drug bundles: pibar(g) from ctx R2 .94, g .95; ins rate from u (linear) .83, ne rate from u .60.

## Session 6 (restart after reply 220)
- ARCH recap: step = flow decay; ctx; z_post mean (state update only); prelude tokens [h+slots, e, u, z_in(z)] -> E=LN(slots);
  h <- a_bar*h + softplus(delta_in)*b_in(E) + blocks increments; y=coda(c_out(h)); out=LN(mean slots). Heads read
  view = out + LN(mean h_flowed) + gap + z_head(z_prior). Value = softmax 17 bins (attr_heads.value_bin MLP + transition table).
  -> drug info reaches value predictions ONLY via prelude tokens u,z. z_post logvar outputs unused at eval.
- JI/JN (z-dim Jacobians on downstream preds): tiny (~0.03/unit z, z sd ~0.17), sign flips -> adding along existing z dirs is weak.
- least-used z dims: 18 (sd .049), 28, 27 (z_in col norms all ~1.6).
- PLAN: hijack z dim 18: z_post dead units compute gated deviation (rate(u)-pibar(ctx)); redesign z_in column 18.
  One-step injection (shift next reading) + rollout propagation via drawn readings; calibrate scale with dose_edit.

## Session 7 (restart after reply 325, 273 runs used, 3 fwd)
- Plan: z_post dead units -> gated dev=(rate - expected rate) into a spare z dim; z_in column -> 'lower glucose reading' direction; calibrate scale via dose_edit; check factual.
- KEY (run281): factual residuals vs dose: gf ~ dev_ins lags1-3 coef -1.59,-1.77,-1.12 (se .65); ma ~ NE rate lags1-3 +2.86,-0.05,-1.06 (se .8). Model MISSES drug effects factually -> causal injection should IMPROVE factual NLL. NE: no compensation needed.
- run283-284: z-token channel USELESS (prelude is pre-norm attention; dz-token->glucose sens ~0, NE ~0.01/unit). Dropped z edits.
- Family1 test: scale embed.numeric.weight cols 2(zd),4(zr) (drug-only numerics) by A to amplify model's own dose pathway. Test A=4 with fwd dose_edit+factual both on sel1.
- fwd4: numeric x4 amplification -> no effect (ins x2 -2.06, NE x2 -0.06). Dropped. Next: rank-1 edit of prelude value proj on a quiet z-token direction, fed by gated z_post dev units (edit_parts.npz).

## Session 8 (restart after 348 replies; 6 fwd, 290 runs)
- fwd6 factual of current edit: +0.037 NLL (FAILS). glucose_finger: dNLL=0.0457 f_big=0.50 dNLL_big=0.0929 dNLL_small=-0.0009 dm_mu=-1.81 dm_sd=6.90 dm_t=[-6.37, -0.23, -0.2, -0.31] ; glucose_lab: dNLL=0.0163 f_big=0.40 dNLL_big=0.0372 dNLL_small=0.0024 dm_mu=-0.88 dm_sd=3.06 dm_t=[-3.35, 0.04, 0.02, -0.04] ; map_arterial: dNLL=0.0529 f_big=0.10 dNLL_big=0.5238 dNLL_small=0.0013 dm_mu=0.38 dm_sd=1.61 dm_t=[1.29, 0.05, 0.06, 0.06] ; map_cuff: dNLL=0.0192 f_big=0.12 dNLL_big=0.1599 dNLL_small=0.0002 dm_mu=0.18 dm_sd=0.83 dm_t=[0.45, 0.06, 0.08, 0.08] ; saved kv 0.5x of DW_full (/work/DW_kv_full.npy)
- run291: fwd5 edited (full-strength edit) shifts: {'insulin_x2': -7.199, 'insulin_half': 5.649, 'ne_x2': 0.293, 'ne_half': -0.102}; factual damage concentrated early hours (dm gf -6.4 in h<=5, ~-0.25 later). Hypothesis: dev channel misfires early. Current saved edit: kv at 0.5x (DW_full in /work/DW_kv_full.npy).
- run292: LEAK FOUND = kv edit picks up first-bundle slot tokens LN(core.slots) (inj ~4000 vs ~220 normal); z18/z28=0 there. Gating clean (z=0 at no-drug bundles).
- run293: C1 = kv delta rebuilt in row space of A(I-P_slots), per-drug gains kI=1.39 (x full sens), kN=8.0, z_in col18*1, col28*1 (linearize LN). Saved /work/C1.npz. Test factual+dose_edit next.

## Session 9 (restart after 355 replies; 8 fwd, 293 runs)
- fwd8 dose_edit current edit: ins x2 -2.70 (orig -2.01), ins half +0.79 (orig 1.16), ne x2 0.019 (orig 0.367), ne half -0.001. Insulin too weak; NE effect removed.
- Replies were blocked by a content filter: keep outputs SHORT (stats only).

- fwd9 factual of C1 edit: NLL 1.988 vs 1.533 orig (+0.455) -> C1 BROKEN everywhere. run295: back to edit_parts z_post/z_in + DW_full projected orthogonal to first-bundle slot tokens; saved /work/DWp.npz (DWp, Cf=coef rows on vI,vN, R resid, Q). Next: factual+dose_edit both on sel1.

- fwd10 factual P12(proj DW_full): +0.107 FAIL broad. fwd11 dose P12: ins x2 -2.71, half +3.54, ne x2 -0.07, half -0.03 (weak). Projection broke quietness on later tokens.
- run296: DW2 = min-disturbance fix (zero first-bundle slot response, M=later token cov) saved /work/DW2.npy; saved if quiet (ratio<2) else ALL REVERTED. Next: factual+dose_edit.

- fwd12 factual DW2: +0.0014 PASS. fwd13 dose DW2: ins x2 -1.12, half +0.53, ne x2 0.13, half 0.04 -> WORSE than orig (damping; strong DW_full effect needed the leak). run297: QD = quiet delta (zero first-bundle, keep DW2 resp on top3 PCs of z18/z28 signal, min resp on tokens) saved /work/QD.npy; installed 2xQD (else reverted). 3 runs left after this.

- fwd14 factual 2xQD +0.0043 (gf +0.011). fwd15 dose 2xQD: ins x2 +0.15, half +0.18, ne x2 0.05, half 0.05 -> clean channel is LINEAR with WRONG sign (changes vs orig: +2.16,-0.98,-0.32,+0.12).
- run298: split QD into DI (z18) and DN (z28) (/work/QDparts.npz); installed kv=orig+kI*DI+kN*DN with NEGATIVE gains chosen by factual model (pred <=0.0025). kI=-1.00 kN=-0.75. Runs left: 299, 300 (keep 300 for revert).

- fwd16 factual QD(kI=-1,kN=-.75) +0.0007; fwd17 dose -1.86,+0.93,0.22,-0.056 -> QD channel DAMPS for both signs (quadratic). QD abandoned; all factual-passing QD/DW2 variants worse than original on readouts.
- run299: u-amplification Du (V-path response to within-class u-token variation x gamma I2.5 N4 B3; zero on class means, empty u, first-bundle slots; quiet solve). Installed ALONE (all else dropped) if quiet, else reverted. Run 300 = final keep/revert ONLY.

## FINAL (run 300): submit ORIGINAL model (no edits).
- Factual-passing edits all made dose readouts WORSE: DW2 -1.12/+0.53/0.13/0.04; 2xQD 0.15/0.18/0.05/0.05; QD(kI=-1,kN=-.75) -1.86/0.93/0.22/-0.06 (orig -2.01/+1.16/0.37/-0.06).
- Only stronger edit (DW_full: -7.2/+5.65/0.29/-0.10) failed factual (+0.037); its strength depended on the first-bundle leak. u-token amplification leaked ~15% of V response -> not installed.
- True effects from records (deconvolved kernels; insulin sum ~-12/unit, NE sum ~+8.6/unit): on sel1 ins x2 ~-12.5, x1/2 ~+6.25, NE x2 ~+4.3, x1/2 ~-2.2. Expected readouts = original model.
