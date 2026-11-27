# Notebook
## Setup facts (runs 1-6)
- Stays ~24h. Readings at integer hours (-1..24); infusion records start at h+0.0833, run 1h. Policy picks hourly rates.
- insulin rate in 0.5 steps (0.5..14, mostly 0.5-3), dose=rate. NE rate 0.01..1.0 (capped at 1.0; 13.5% at cap), dose=4.8*rate (weight 80kg for all).
- 61% stays get insulin, 43% NE. stay_id sequential, no structure.
- Dose pathway in model: embed.numeric cols z_dose/z_rate (cols 2..5), embed.bins.dose/rate -> encode_action -> u_k -> prelude tokens + z_post input.

## Insulin policy (runs 9-12)
- Deterministic sliding scale on latest glucose (fresh gf, else fresh gl): none if <150; band base rates ~1 [150,180), ~2 [180,220), ~3 [220,250), ~3.7 [250,300) (has inner step), ~6.5 >=300.
- Within band, rate noise (0.5 steps) is independent of all observed covariates (R2~1e-4) => randomized experiment. No hidden confounding apparent.
- gf measured every hour while last glucose >150.

## NE policy (runs 13-14)
- Uses map_arterial if present else cuff. Start (~0.05) when MAP<65. Titration: log(rate_t/rate_{t-1}) = adj + N(0,~0.3); adj ~ +0.1 (<65), 0 (65-74), -0.25 (>=75, sometimes stop). Cap 1.0.
- So NE rate = accumulated random walk => lots of exogenous variation.
- Insulin rate ~ base(band)*exp(N(0,0.3)) rounded to 0.5 (fits band-1 shares).

## Insulin effect estimate (runs 15-18) -- randomization within band, deconvolved for policy feedback
- 1h effect -2.0 mg/dL per U/h, same in all bands (linear, no glucose dependence). gf and gl same glucose (no offset, diff sd 9).
- Structural impulse K(k), k=1..10 (fixed other doses): -2.0 -2.5 -2.2 -1.8 -1.4 -1.15 -0.94 -0.75 -0.50 -0.42 (sum -13.7).
- Insulin does not move NE or MAP materially.

## NE effect estimate (runs 19-20)
- NE noise (rate - E[rate|prev rate, MAP band]) unpredictable (R2 0.001). Stops only from rate<=0.06 & MAP>=75.
- Structural K_N(k) on MAP per (mcg/kg/min * 1h): +3.6 +2.7 +1.7 +1.1 +0.4 +0.45 ~0 ... (sum ~10.4). Same on ma and mc. mc reads ~3 mmHg above ma (sd 6.7).
- NE first stage persists (0.73,0.56,0.44,...) due to random-walk titration. NE->glucose maybe +0.4 (noisy, not scored).

## Model diagnostics (fwd 1-2, run 25-27), tuning sel1 (100 stays, /work/sel1.json)
- Original dose_edit: ins x2 -2.14, x1/2 +1.26, NE x2 +1.08, x1/2 -0.73.  My record-based truth on same reached rows: -14.1, +7.0, +4.7, -2.4.
- Model shift grows ~linearly with lag (-0.55 lag1 -> -3.5 lag12); truth -3.0 -> -19 (concave).
- Factual baseline value NLL 1.52793 (5238 values). Model factual within-band slope: insulin -0.45/U/h (data -2.0), NE +0.94 (data 3.6) => model under-uses doses even teacher-forced; amplifying dose effect should improve NLL.
- Readout rows = recorded outcome readings (gf / map_arterial) at hours 13-24, 'reached' = after a drug record at hour>=12.

## Restart 2 (runs 37-46)
- ARCH recap: atom emb = code+kind+ekind+categorical+bins(value/dose/rate, tables shared across codes)+numeric([z_val,has_val,z_dose,has_dose,z_rate,has_rate]).
  z = (x-median)/iqr clip +-10. Insulin rate/dose median 1.5 iqr 1.0, bins saturate at >=5 (bin16). NE rate median .46 iqr .6, bins saturate >=1.0 (cap).
  u_k = LN(mlp2(sum mlp1(atoms)/sqrt n + size)). z_post([ctx,e,u]) 576->192->64 (GELU). prelude tokens [h+slots,e,u,z_in(z)], E=prelude_norm(slots).
  h <- a_bar h + softplus(delta) b_in(E) + (blocks(h+E)-(h+E)); h=coda(c_out(h)); c_out NOT ~identity. out=out_norm(mean slots).
  Heads: view=out+out_norm(mean h_flowed)+gap(log dt)+z_head(z~prior); q=query_norm(view+type_emb[type]);
  with_code=q+code emb+last_numeric(8 cols)+bins.value[last bin]+bins.rate[last rate bin]+last_status; value logits=MLP(192-192-17)(with_code)+embed_transition.value_bin[code*17+lastbin].
- NOTE: variable g (train grid) was overwritten in run 37; gg (train grid with lags, glu_now, band, e_ins, map_pol, nor_prev, e_nor) still exists.
- Policy fn pfun(glucose): smoothed mean insulin rate by latest glucose (train). p jumps 0->0.5 at 150, ~1 at 160, ~2 at 200, ~4 at 275, ~5.3 at 300.
- Tuning residual regression (n=1297 gf, noisy): resid on e1 -0.95+-.63, p1 -0.59+-.56 (inconclusive). Model pred-mean slope on e1,e2: -0.5, -1.3 (+-.35); observed -1.5, -2.0 (+-.7).
- States (fwd_3, train selT) at insulin bundles: out decodes p R2 .93, ins .80, e=ins-p .46; z decodes ins .73, p .74, e .32. NE: out nor .80, prev .71, e .26.
  => z is NOT a clean deviation channel; state encodes band well, dose coarsely.
- BELIEF: model fits factual well => missing insulin effect partly absorbed into glucose-level dynamics; naive dose amplification would double count
  (est. bias ~10 mg/dL if x4). PLAN: gated linear readout in value head: repurpose a low-importance hidden unit of attr_heads.value_bin, gated on code
  (gf/gl for insulin, ma/mc for NE), input = linear decoder of dose deviation from q, output weights = logit tilt along bin centres; compensation of
  last-glucose-level effects via embed_transition.value_bin. Need exact head re-implementation -> factual on selT (same stays as fwd_3 states).

## Restart 3 (runs 72-81)
- ARCH re-read: head 'last' register = last value of SAME code (no dose in head). Head z drawn from PRIOR 4x, probs averaged (explains my head NLL gap, run 71).
  core: a_bar 0.31-0.46 per bundle; h rewritten each step by prelude/blocks/coda (pre-norm transformer layers, 4 heads, ff 768); no dead h dims. => no cheap dedicated memory channel.
- Bins: gf centres 72..271 (top bin >=260 -> centre 271), ma 33..101 (>=98). insulin rate/dose bins saturate >5; NE rate bin 16 for >1.0.
- One-step OLS gf_t on gf lags1-3, ins lags1-10, pol lags1-4 (train n=228k): AR .48/.30/.21 (sum .98); dose -1.99 -1.47 -0.89 +.17 +.81 +.65 +.40 +.32 +.23 +.03.
  Model predicted_mean (fwd_4 selT, n=1704, resid sd 3.2): AR .46/.24/.17 (sum .87, const 18); dose -0.31 -0.10 -0.49 -0.04 +.51 +.31 +.32 +.31 +.05 +.10.
  => model misses short-lag insulin effect (lags1-3), compensates with stronger mean reversion.
- Linear surrogate rollout (sel1, x2): model-coef surrogate -1.2 (tool -2.14), but shape wrong at late lags (surrogate rebounds, tool grows ~linearly -0.55 -> -3.53).
  => linear AR(3) projection is NOT a reliable rollout model. Model propagates drawn shifts ~fully.
- DECISION: build local full re-implementation (core step verified vs fwd_3 states), then local rollouts & counterfactual states.

## Restart 4 (runs 94-119): exact encoder in /work/impl.py (exec it)
- Atom emb: code+kind+ekind+unit[row]+rate_unit[row]+kq[0]+bins(value,dose,rate; side=left)+numeric. Unit rows: gf 0, gl 5, MAP 2, insulin 3, NE 4;
  rate_unit: insulin 2, NE 3. END atoms (kind 2): unit 0, rate_unit 0, NO rate/dose. END atoms routed to e (encode); START drug atoms to u.
- Empty sets: e/u = norm(mlp2(size(log1p 0))) (NOT zero). z_post input [out_prev+out_norm(mean hf)+gap(log(dt_s)/10), e, u], gelu everywhere.
- Verified: z posterior mean matches fwd_3 states exactly (abs err ~0) on all bundle kinds (selT train stays).

## Restart 5 (runs 134-141): EXACT CORE STEP FOUND (code in /work/core_test.py, needs impl.py exec'd first)
- fwd_3 states rows are NOT sorted like my bundle table: align by (stay_id, round(hour*60)) -> SxA = Sx[perm_S].
- core step exact (out err 1e-5, h 4e-4): memory M = e of ALL bundles of the stay up to AND INCLUDING current (last 128),
  age = (t_now - t_j) in SECONDS, agee = core.age(log1p(age)) ADDED TO M BEFORE kv projection (age_mode 'inadd': keys and values),
  u of bundles w/o drug = encoder empty-set vector (NOT zero). ctx gap = gap(log(dt_s)/10). z = posterior mean (first 32 of z_post).
- Lesson: kernel functions have no source -> keep all code in /work/*.py files.

- HEAD EXACT (runs 144-148, code /work/head.py): view = out + out_norm(mean hf) + gap(log(dt_s)/10) + z_head(z), z~prior (z_prior out[:32]=mu, [32:]=log-sd);
  q = query_norm(view + type_emb[1]) (type 1 for measurement values); last_numeric 8 cols for measurements: [has_last,0,0,log1p(age_s)/10,0,0,z_last,0];
  last_status row 0 (zero) for measurements; bins.rate[0]=0. logits = value_bin MLP(gelu) + embed_transition.value_bin[code*17+last_bin]. pm = sum_{b=1..16} p_b*centre[b-1].
  Matches tool up to 4-draw z noise (pm RMSE ~1.3-1.8).

- first bundle: h0=0,out0=0, gap with dt=1s. ROLLOUT ENGINE /work/rollout.py (exec impl.py, core_test.py, head.py first):
  R=prep_rollout(df,ids); res=run_rollout(R,W,arms,n_paths,seed); rows,reached,o=readout(R,res,arms,drug,outcome).
  Validated vs tool fwd_1 (sel1 first 30 stays): ins x2 -2.90 (tool -2.81), x1/2 1.48 (1.70), NE x2 1.09 (1.15), x1/2 -.36 (-.33); per-row corr .93-.96.
  Draws = bin centre fed back; 4 prior-z draws averaged; common random numbers. 30 stays x5 arms x20 paths = 14 s.

## Restart 5 (runs 134-141): EXACT CORE STEP FOUND (code in /work/core_test.py, needs impl.py exec'd first)
- fwd_3 states rows are NOT sorted like my bundle table: align by (stay_id, round(hour*60)) -> SxA = Sx[perm_S].
- core step exact (out err 1e-5, h 4e-4): memory M = e of ALL bundles of the stay up to AND INCLUDING current (last 128),
  age = (t_now - t_j) in SECONDS, agee = core.age(log1p(age)) ADDED TO M BEFORE kv projection (age_mode 'inadd': keys and values),
  u of bundles w/o drug = encoder empty-set vector (NOT zero). ctx gap = gap(log(dt_s)/10). z = posterior mean (first 32 of z_post).
- Lesson: kernel functions have no source -> keep all code in /work/*.py files.

## Oracle lag-only kernels (run 207), ids_o 100 train stays, 20 paths
g-1.0: {'ins': 2.0} pred -9.71 true -16.51; {'ins': 0.5} pred 5.15 true 8.26; {'ins': 0.0} pred 8.15 true 16.51; {'ins': 3.0} pred -17.28 true -33.03; {'ins': 2.0, 'ins_from': 16.0} pred -6.98 true -14.19
g-1.5: {'ins': 2.0} pred -13.09 true -16.51; {'ins': 0.5} pred 6.68 true 8.26; {'ins': 0.0} pred 10.98 true 16.51; {'ins': 3.0} pred -24.09 true -33.03; {'ins': 2.0, 'ins_from': 16.0} pred -9.31 true -14.19
g-1.2,-0.4: {'ins': 2.0} pred -13.31 true -16.51; {'ins': 0.5} pred 6.84 true 8.26; {'ins': 0.0} pred 11.21 true 16.51; {'ins': 3.0} pred -24.57 true -33.03; {'ins': 2.0, 'ins_from': 16.0} pred -9.37 true -14.19

## Restart 7 (runs 266-290)
- Head-only readouts (gated MLP unit + logit tilt) FAIL: state's dose signal entangled with glucose level; within 0.005-nat budget only ~20% of one-step target (D4 teacher-forced arms, 100 train stays). Global tilt explodes tails; virtual last-value shift too weak (0.14/mgdl).
- Channel design (chan.py) too big to finish. NEXT: amplify encoder dose numerics (z_dose,z_rate cols) by k, judge with tool on sel1.
- fwd5/6 (sel1 tuning): numeric z_dose/z_rate cols x3 -> factual NLL 1.52793->1.52582 (better); ins x2 -2.22 (orig -2.14), x1/2 +1.39; NE x2 +2.66 (orig 1.08, truth 4.7), x1/2 -1.48 (truth -2.4).
- cand2: + bins.rate/dose rows1-16 deviations x3 (ref mean) to reach insulin path.
- cand2 (bins x3) BAD: factual +0.0055 (MAP worse), ins x2 -2.43, NE x2 1.56/x.5 -2.73 -> reverted.
- cand3: numeric x5 + head gated readout unit 109 (Q_build, mu=.3 resid-target, lam=.5, kap4 G20, bounded tilt clip60/Vb201), w in /work/w_ins_head.npy.

## Restart 8 (run 294): diagnosis of fwd_9 (cand3 factual, edited only, 1.57204)
- (['fwd_2', 'factual_original.csv'], 'n', 100, 'ov', 100, 'rows', 5238, 'nll 1.52793', {'glucose_finger': 1.5591, 'glucose_lab': 1.2271, 'map_arterial': 1.3894, 'map_cuff': 1.7687})
- (['fwd_4', 'factual_original.csv'], 'n', 100, 'ov', 0, 'nll(all) 1.42354')
- (['fwd_5', 'factual_edited.csv'], 'n', 100, 'ov', 100, 'rows', 5238, 'nll 1.52582', {'glucose_finger': 1.5571, 'glucose_lab': 1.2247, 'map_arterial': 1.3873, 'map_cuff': 1.7666})
- (['fwd_5', 'factual_original.csv'], 'n', 100, 'ov', 100, 'rows', 5238, 'nll 1.52793', {'glucose_finger': 1.5591, 'glucose_lab': 1.2271, 'map_arterial': 1.3894, 'map_cuff': 1.7687})
- (['fwd_7', 'factual_edited.csv'], 'n', 100, 'ov', 100, 'rows', 5238, 'nll 1.53339', {'glucose_finger': 1.5582, 'glucose_lab': 1.2204, 'map_arterial': 1.3964, 'map_cuff': 1.786})
- (['fwd_7', 'factual_original.csv'], 'n', 100, 'ov', 100, 'rows', 5238, 'nll 1.52793', {'glucose_finger': 1.5591, 'glucose_lab': 1.2271, 'map_arterial': 1.3894, 'map_cuff': 1.7687})
- (['fwd_9', 'factual_edited.csv'], 'n', 100, 'ov', 100, 'rows', 5238, 'nll 1.57204', {'glucose_finger': 1.6592, 'glucose_lab': 1.324, 'map_arterial': 1.3886, 'map_cuff': 1.7672})

- fwd_9 = sel1: cand3 factual 1.57204 vs orig 1.52793 (+0.044; gf 1.559->1.659, gl 1.227->1.324, MAP ok). fwd_10 cand3 dose_edit sel1: ins x2 -3.68, x.5 +3.34, NE x2 2.82, x.5 -2.22.
- DECISION run 295: head readout dropped (too costly, too weak). Kept numeric cols 2,4 x5. Local scan of numeric (k2,k4) arms [fact, ins2, ins.5, ne2, ne.5, ne0, ne3, ne1.5] on sel1[:30]:
  (1, 1): (('insulin', {0: '[[119.09 120.61 134.72 ... 118.77 128.07 134.92]\n [123.57 13', 1: '[[119.09 120.61 134.72 ... 118.77 128.07 134.92]\n [123.72 13', 2: '[[119.09 120.61 134.72 ... 118.77 128.07 134.92]\n [123.57 13', 3: '[[119.09 120.61 134.72 ... 118.77 128.07 134.92]\n [123.57 13', 4: '[[119.09 120.61 134.72 ... 118.77 128.07 134.92]\n [123.57 13', 5: '[[119.09 120.61 134.72 ... 118.77 128.07 134.92]\n [123.57 13', 6: '[[119.09 120.61 134.72 ... 118.77 128.07 134.92]\n [123.57 13', 7: '[[119.09 120.61 134.72 ... 118.77 128.07 134.92]\n [123.57 13'}), ('norepinephrine', {0: '[[63.65 64.44 62.5  ... 62.73 65.13 65.14]\n [63.95 62.35 68.', 1: '[[63.65 64.44 62.5  ... 62.73 65.13 65.14]\n [63.95 62.35 68.', 2: '[[63.65 64.44 62.5  ... 62.73 65.13 65.14]\n [63.95 62.35 68.', 3: '[[63.65 64.44 62.5  ... 62.73 65.13 65.14]\n [63.95 62.35 68.', 4: '[[63.65 64.44 62.5  ... 62.73 65.13 65.14]\n [63.95 62.35 68.', 5: '[[63.65 64.44 62.5  ... 62.73 65.13 65.14]\n [63.95 62.35 68.', 6: '[[63.65 64.44 62.5  ... 62.73 65.13 65.14]\n [63.95 62.35 68.', 7: '[[63.65 64.44 62.5  ... 62.73 65.13 65.14]\n [63.95 62.35 68.'}))
  (5, 5): (('insulin', {0: '[[119.93 120.43 134.62 ... 119.09 127.93 134.78]\n [123.67 13', 1: '[[119.93 120.43 134.62 ... 119.09 127.93 134.78]\n [123.65 13', 2: '[[119.93 120.43 134.62 ... 119.09 127.93 134.78]\n [123.72 13', 3: '[[119.93 120.43 134.62 ... 119.09 127.93 134.78]\n [123.67 13', 4: '[[119.93 120.43 134.62 ... 119.09 127.93 134.78]\n [123.67 13', 5: '[[119.93 120.43 134.62 ... 119.09 127.93 134.78]\n [123.67 13', 6: '[[119.93 120.43 134.62 ... 119.09 127.93 134.78]\n [123.67 13', 7: '[[119.93 120.43 134.62 ... 119.09 127.93 134.78]\n [123.67 13'}), ('norepinephrine', {0: '[[63.17 63.9  62.01 ... 62.29 64.66 64.58]\n [63.49 61.91 68.', 1: '[[63.17 63.9  62.01 ... 62.29 64.66 64.58]\n [63.49 61.91 68.', 2: '[[63.17 63.9  62.01 ... 62.29 64.66 64.58]\n [63.49 61.91 68.', 3: '[[63.17 63.9  62.01 ... 62.29 64.66 64.58]\n [63.49 61.91 68.', 4: '[[63.17 63.9  62.01 ... 62.29 64.66 64.58]\n [63.49 61.91 68.', 5: '[[63.17 63.9  62.01 ... 62.29 64.66 64.58]\n [63.49 61.91 68.', 6: '[[63.17 63.9  62.01 ... 62.29 64.66 64.58]\n [63.49 61.91 68.', 7: '[[63.17 63.9  62.01 ... 62.29 64.66 64.58]\n [63.49 61.91 68.'}))
  (8, 8): (('insulin', {0: '[[120.15 120.65 134.88 ... 118.78 128.17 135.05]\n [123.71 13', 1: '[[120.15 120.65 134.88 ... 118.78 128.17 135.05]\n [123.5  13', 2: '[[120.15 120.65 134.88 ... 118.78 128.17 135.05]\n [123.89 13', 3: '[[120.15 120.65 134.88 ... 118.78 128.17 135.05]\n [123.71 13', 4: '[[120.15 120.65 134.88 ... 118.78 128.17 135.05]\n [123.71 13', 5: '[[120.15 120.65 134.88 ... 118.78 128.17 135.05]\n [123.71 13', 6: '[[120.15 120.65 134.88 ... 118.78 128.17 135.05]\n [123.71 13', 7: '[[120.15 120.65 134.88 ... 118.78 128.17 135.05]\n [123.71 13'}), ('norepinephrine', {0: '[[63.57 64.34 62.41 ... 62.66 65.05 65.03]\n [63.59 62.   68.', 1: '[[63.57 64.34 62.41 ... 62.66 65.05 65.03]\n [63.59 62.   68.', 2: '[[63.57 64.34 62.41 ... 62.66 65.05 65.03]\n [63.59 62.   68.', 3: '[[63.57 64.34 62.41 ... 62.66 65.05 65.03]\n [63.59 62.   68.', 4: '[[63.57 64.34 62.41 ... 62.66 65.05 65.03]\n [63.59 62.   68.', 5: '[[63.57 64.34 62.41 ... 62.66 65.05 65.03]\n [63.59 62.   68.', 6: '[[63.57 64.34 62.41 ... 62.66 65.05 65.03]\n [63.59 62.   68.', 7: '[[63.57 64.34 62.41 ... 62.66 65.05 65.03]\n [63.59 62.   68.'}))
  (5, 10): (('insulin', {0: '[[120.02 120.52 134.67 ... 119.17 128.01 134.85]\n [123.69 13', 1: '[[120.02 120.52 134.67 ... 119.17 128.01 134.85]\n [123.44 12', 2: '[[120.02 120.52 134.67 ... 119.17 128.01 134.85]\n [123.86 13', 3: '[[120.02 120.52 134.67 ... 119.17 128.01 134.85]\n [123.69 13', 4: '[[120.02 120.52 134.67 ... 119.17 128.01 134.85]\n [123.69 13', 5: '[[120.02 120.52 134.67 ... 119.17 128.01 134.85]\n [123.69 13', 6: '[[120.02 120.52 134.67 ... 119.17 128.01 134.85]\n [123.69 13', 7: '[[120.02 120.52 134.67 ... 119.17 128.01 134.85]\n [123.69 13'}), ('norepinephrine', {0: '[[63.35 64.08 62.18 ... 62.46 64.83 64.78]\n [63.48 61.9  68.', 1: '[[63.35 64.08 62.18 ... 62.46 64.83 64.78]\n [63.48 61.9  68.', 2: '[[63.35 64.08 62.18 ... 62.46 64.83 64.78]\n [63.48 61.9  68.', 3: '[[63.35 64.08 62.18 ... 62.46 64.83 64.78]\n [63.48 61.9  68.', 4: '[[63.35 64.08 62.18 ... 62.46 64.83 64.78]\n [63.48 61.9  68.', 5: '[[63.35 64.08 62.18 ... 62.46 64.83 64.78]\n [63.48 61.9  68.', 6: '[[63.35 64.08 62.18 ... 62.46 64.83 64.78]\n [63.48 61.9  68.', 7: '[[63.35 64.08 62.18 ... 62.46 64.83 64.78]\n [63.48 61.9  68.'}))
  (10, 5): (('insulin', {0: '[[120.15 120.65 134.87 ... 118.78 128.16 135.03]\n [123.7  13', 1: '[[120.15 120.65 134.87 ... 118.78 128.16 135.03]\n [123.59 13', 2: '[[120.15 120.65 134.87 ... 118.78 128.16 135.03]\n [123.87 13', 3: '[[120.15 120.65 134.87 ... 118.78 128.16 135.03]\n [123.7  13', 4: '[[120.15 120.65 134.87 ... 118.78 128.16 135.03]\n [123.7  13', 5: '[[120.15 120.65 134.87 ... 118.78 128.16 135.03]\n [123.7  13', 6: '[[120.15 120.65 134.87 ... 118.78 128.16 135.03]\n [123.7  13', 7: '[[120.15 120.65 134.87 ... 118.78 128.16 135.03]\n [123.7  13'}), ('norepinephrine', {0: '[[63.62 64.39 62.47 ... 62.71 65.1  65.09]\n [63.67 62.08 68.', 1: '[[63.62 64.39 62.47 ... 62.71 65.1  65.09]\n [63.67 62.08 68.', 2: '[[63.62 64.39 62.47 ... 62.71 65.1  65.09]\n [63.67 62.08 68.', 3: '[[63.62 64.39 62.47 ... 62.71 65.1  65.09]\n [63.67 62.08 68.', 4: '[[63.62 64.39 62.47 ... 62.71 65.1  65.09]\n [63.67 62.08 68.', 5: '[[63.62 64.39 62.47 ... 62.71 65.1  65.09]\n [63.67 62.08 68.', 6: '[[63.62 64.39 62.47 ... 62.71 65.1  65.09]\n [63.67 62.08 68.', 7: '[[63.62 64.39 62.47 ... 62.71 65.1  65.09]\n [63.67 62.08 68.'}))
  (2, 8): (('insulin', {0: '[[120.28 128.57 135.1  ... 118.92 128.36 135.27]\n [123.67 13', 1: '[[120.28 128.57 135.1  ... 118.92 128.36 135.27]\n [123.56 13', 2: '[[120.28 128.57 135.1  ... 118.92 128.36 135.27]\n [123.69 13', 3: '[[120.28 128.57 135.1  ... 118.92 128.36 135.27]\n [123.67 13', 4: '[[120.28 128.57 135.1  ... 118.92 128.36 135.27]\n [123.67 13', 5: '[[120.28 128.57 135.1  ... 118.92 128.36 135.27]\n [123.67 13', 6: '[[120.28 128.57 135.1  ... 118.92 128.36 135.27]\n [123.67 13', 7: '[[120.28 128.57 135.1  ... 118.92 128.36 135.27]\n [123.67 13'}), ('norepinephrine', {0: '[[63.41 64.19 62.27 ... 62.52 64.89 64.86]\n [63.72 62.13 68.', 1: '[[63.41 64.19 62.27 ... 62.52 64.89 64.86]\n [63.72 62.13 68.', 2: '[[63.41 64.19 62.27 ... 62.52 64.89 64.86]\n [63.72 62.13 68.', 3: '[[63.41 64.19 62.27 ... 62.52 64.89 64.86]\n [63.72 62.13 68.', 4: '[[63.41 64.19 62.27 ... 62.52 64.89 64.86]\n [63.72 62.13 68.', 5: '[[63.41 64.19 62.27 ... 62.52 64.89 64.86]\n [63.72 62.13 68.', 6: '[[63.41 64.19 62.27 ... 62.52 64.89 64.86]\n [63.72 62.13 68.', 7: '[[63.41 64.19 62.27 ... 62.52 64.89 64.86]\n [63.72 62.13 68.'}))

- run 295: head edits DROPPED (edited = embed.numeric only, x5). VOCAB: z=(x-median)/iqr clipped +-10 (linear). NE arm key is "ne" (not nor).
- run 296: local scan sel1 100 stays, 10 paths, (k2,k4) numeric cols 2,4: (1, 1): {'ins': {'i2': -2.18, 'i.5': 1.09, 'i0': 1.09}, 'ne': {'n2': 1.19, 'n.5': -0.73, 'n0': -0.24, 'n3': 2.27, 'n1.5': 0.74}, 'cuff': {'n2': 0.93}, 'lab': {'i2': -1.78}, 'relerr': 3.829}; (3, 3): {'ins': {'i2': -2.19, 'i.5': 1.2, 'i0': 1.08}, 'ne': {'n2': 2.76, 'n.5': -1.44, 'n0': -0.92, 'n3': 4.08, 'n1.5': 1.71}, 'cuff': {'n2': 2.41}, 'lab': {'i2': -2.01}, 'relerr': 2.443}; (5, 5): {'ins': {'i2': -1.81, 'i.5': 1.45, 'i0': 1.61}, 'ne': {'n2': 2.85, 'n.5': -2.18, 'n0': -1.6, 'n3': 3.98, 'n1.5': 1.98}, 'cuff': {'n2': 2.5}, 'lab': {'i2': -1.61}, 'relerr': 1.86}; (7, 7): {'ins': {'i2': -1.81, 'i.5': 1.61, 'i0': 1.72}, 'ne': {'n2': 2.68, 'n.5': -2.74, 'n0': -2.21, 'n3': 3.6, 'n1.5': 1.73}, 'cuff': {'n2': 2.31}, 'lab': {'i2': -1.7}, 'relerr': 2.006}; (10, 10): {'ins': {'i2': -1.81, 'i.5': 1.61, 'i0': 1.8}, 'ne': {'n2': 2.37, 'n.5': -3.17, 'n0': -2.8, 'n3': 3.13, 'n1.5': 1.58}, 'cuff': {'n2': 2.0}, 'lab': {'i2': -1.68}, 'relerr': 2.244}; (5, 10): {'ins': {'i2': -1.94, 'i.5': 1.79, 'i0': 2.05}, 'ne': {'n2': 2.86, 'n.5': -2.68, 'n0': -2.08, 'n3': 3.81, 'n1.5': 1.85}, 'cuff': {'n2': 2.49}, 'lab': {'i2': -1.85}, 'relerr': 1.897}; (10, 5): {'ins': {'i2': -1.66, 'i.5': 1.58, 'i0': 1.58}, 'ne': {'n2': 2.5, 'n.5': -3.32, 'n0': -2.83, 'n3': 3.39, 'n1.5': 1.68}, 'cuff': {'n2': 2.13}, 'lab': {'i2': -1.47}, 'relerr': 2.203}
  BEST (relerr vs NE truth 4.7/-2.35/-4.7/9.4/2.35) = (5, 5) SAVED. local TF nll {'orig': 1.5233968764190775, 'best': 1.5215948505534067}

- run 296 scan (local=tool within ~0.1): (1,1) ne2 1.19 ne.5 -.73; (3,3) 2.76/-1.44; (5,5) 2.85/-2.18 ne0 -1.6 ne3 3.98; (7,7) 2.68/-2.74; (10,10) 2.37/-3.17 -> NE x2 HUMP (confounding in repr). ins x2 ~-1.8..-2.2 for all k (no insulin lever).
- run 297: score = sum rel err over NE(5 arms)+cuff+ins(3)+lab vs truth est; scan (k2 low,k4 high); results: (1, 1): ne {'n2': 1.19, 'n.5': -0.73, 'n0': -0.24, 'n3': 2.27, 'n1.5': 0.74} ins {'i2': -2.18, 'i.5': 1.09, 'i0': 1.09} sc 8.193 dtf None; (3, 3): ne {'n2': 2.76, 'n.5': -1.44, 'n0': -0.92, 'n3': 4.08, 'n1.5': 1.71} ins {'i2': -2.19, 'i.5': 1.2, 'i0': 1.08} sc 6.464 dtf -0.00188; (5, 5): ne {'n2': 2.85, 'n.5': -2.18, 'n0': -1.6, 'n3': 3.98, 'n1.5': 1.98} ins {'i2': -1.81, 'i.5': 1.45, 'i0': 1.61} sc 5.847 dtf -0.0018; (7, 7): ne {'n2': 2.68, 'n.5': -2.74, 'n0': -2.21, 'n3': 3.6, 'n1.5': 1.73} ins {'i2': -1.81, 'i.5': 1.61, 'i0': 1.72} sc 6.003 dtf -0.00102; (10, 10): ne {'n2': 2.37, 'n.5': -3.17, 'n0': -2.8, 'n3': 3.13, 'n1.5': 1.58} ins {'i2': -1.81, 'i.5': 1.61, 'i0': 1.8} sc 6.302 dtf 0.00139; (5, 10): ne {'n2': 2.86, 'n.5': -2.68, 'n0': -2.08, 'n3': 3.81, 'n1.5': 1.85} ins {'i2': -1.94, 'i.5': 1.79, 'i0': 2.05} sc 5.796 dtf -0.00112; (10, 5): ne {'n2': 2.5, 'n.5': -3.32, 'n0': -2.83, 'n3': 3.39, 'n1.5': 1.68} ins {'i2': -1.66, 'i.5': 1.58, 'i0': 1.58} sc 6.273 dtf 0.00155; (1, 10): ne {'n2': 3.41, 'n.5': -1.81, 'n0': -1.28, 'n3': 4.56, 'n1.5': 2.15} ins {'i2': -2.03, 'i.5': 1.86, 'i0': 2.26} sc 5.588 dtf -0.00229; (2, 10): ne {'n2': 3.3, 'n.5': -2.01, 'n0': -1.42, 'n3': 4.43, 'n1.5': 2.16} ins {'i2': -1.93, 'i.5': 1.82, 'i0': 2.17} sc 5.548 dtf -0.00203; (3, 10): ne {'n2': 3.17, 'n.5': -2.22, 'n0': -1.66, 'n3': 4.25, 'n1.5': 2.09} ins {'i2': -1.9, 'i.5': 1.79, 'i0': 2.12} sc 5.522 dtf -0.00175; (2, 15): ne {'n2': 2.91, 'n.5': -2.9, 'n0': -2.26, 'n3': 3.72, 'n1.5': 1.95} ins {'i2': -1.81, 'i.5': 2.06, 'i0': 2.52} sc 5.761 dtf 0.00139; (3, 15): ne {'n2': 2.83, 'n.5': -2.97, 'n0': -2.38, 'n3': 3.64, 'n1.5': 1.9} ins {'i2': -1.92, 'i.5': 1.94, 'i0': 2.41} sc 5.835 dtf 0.00146; (1, 15): ne {'n2': 2.97, 'n.5': -2.82, 'n0': -2.19, 'n3': 3.77, 'n1.5': 1.96} ins {'i2': -1.66, 'i.5': 2.14, 'i0': 2.6} sc 5.701 dtf 0.00125
  BEST saved = (3, 10). sel2.json = 100 other tuning stays for factual check.

## FINAL (run 298) - edit hash eb1bf45c92
- Final edit: embed.numeric.weight col 2 (z_dose) x3, col 4 (z_rate) x10; all else original (head readout cand3 dropped: +0.044 nats, only -1.5 extra insulin effect).
- Tool factual (paired, tuning): sel1 1.52793->1.52579 (-0.0021); sel2 1.62820->1.62797 (-0.0002); sel3 (ids 156-270) 1.66225->1.66366 (+0.0014). Pooled ~-0.0004. Budget +0.005 safe.
- Tool dose_edit sel1 (84 ins / 56 NE stays): orig ins x2 -2.14, x.5 +1.26, NE x2 +1.08, x.5 -0.73 -> edited ins x2 -1.92, x.5 +1.98, NE x2 +3.13, x.5 -2.30.
  sel2 (39/13 stays): orig -4.18/+2.23/+1.26/-0.95 -> edited -4.46/+2.63/+4.08/-2.62.
- Local engine (sel1, validated vs tool within ~0.1): NE x0 -1.66, x3 +4.25, x1.5 +2.09, map_cuff x2 +2.77; insulin x0 +2.12, glucose_lab x2 -1.88.
- Beliefs: record-based effect estimates (one-step OLS lag kernel with glucose lags + policy controls, propagated): insulin x2 ~ -16.5 mg/dL (linear in dose change),
  NE x2 ~ +4.7, x.5 ~ -2.35 mmHg. Model's insulin pathway cannot be strengthened cheaply: dose info enters only via START-atom u-set, decays in recurrent state;
  amplifying encoder dose inputs does not change glucose response; head readouts are entangled with glucose band (factual cost too high).
  NE response saturates (hump) for large amplification: x2 peaks near k4~10 with low k2; x0/x3 under-predicted.
- STATUS: DONE, submit.
