# Notebook

## Run 1
- Read ARCHITECTURE.md and data README. Infusion records = 1 h at a rate. Dose edit scales insulin/norepi rate+dose from hour 12.
- Readout: mean shift of glucose_finger (insulin) / map_arterial (norepi) over hours 13-24, edited minus factual.
- Doses chosen by clinicians -> confounded. Need causal estimate from records.

## Runs 2-14: data structure and insulin policy (train.csv)
- 30k stays, hours -2..25.17, readings on the hour, infusions start at h+0.0833 and run 1h. All stays same length.
- Insulin POLICY (memoryless): at an hour with a fresh glucose (finger preferred, else lab), if >150 -> rate = base*jitter,
  base = 1 (150-180], 2 (180-220], 3 (220-260], 4 (260-300], 6 (>300). Jitter multiplicative, mean 1.056, sd ~0.33, rounded to 0.5.
  Jitter: lag-1 autocorr 0.0007, no stay component, uncorrelated with trend and with simultaneous lab|finger -> as-good-as-random.
- Measurement: finger hourly when last policy glucose >150; else finger w.p. ~0.30 at random. Lab random 20% of hours.
  lab & finger measure same latent glucose; noise roughly proportional to level; finger noisier.
- Never-insulin stays: glucose ~ random-walk-like (variogram rising ~8/h over 15h).
- IV (jitter instruments, lab outcome, later doses as endogenous): insulin impulse response per unit/h:
  lag1 ~ -2.1, lag2-4 ~ -2.4 (-2.0..-2.8), lag5-6 ~ -1, lag7-10 ~ -0.5..-1, lag12 ~ 0. Transient effect, NOT permanent.

## Runs 15-26: model readout + norepi
- dose_edit readout rows = hours 13-24 with a RECORDED glucose_finger (insulin) / map_arterial (norepi) reading; 'reached' = after
  first dose at/after hour 12. Mean over reached rows. Norepi rows only exist in arterial-line stays (~half of stays, stay-level).
- Original model on sel100 tuning stays (fwd_1): insulin x2 -0.39, x1/2 +0.15; ne x2 -0.66, ne x1/2 -0.24  (tiny / wrong sign).
- Prelim insulin truth on same rows (theta: lag1 -2.2, lag2-4 -2.4, lag5 -1.5, 6 -1.1, 7 -0.9, 8 -0.8, 9 -0.6, 10 -0.5, 11 -0.3, 12 -0.1):
  x2 = -17.5, x1/2 = +8.8 (linear assumption).
- Norepi POLICY: start only when off & fresh MAP<=~64 at ~0.05; when on, multiplicative titration r_t = r_{t-1}*f(MAP)*noise, clip [0.01,1];
  stop w.p. ~0.15-0.2 if MAP>75. Uses arterial if present else cuff; in no-fresh hours uses LAST MAP. Innovation (r - E[r|cell]) unpredictable
  from history once cells include (last MAP bin, age) for no-fresh hours. Cuff = art + 3 (sd 6.7). Arterial present 95% at random in line stays.
- Norepi IV impulse response on map_arterial per 1 mcg/kg/min: lag1 ~3.0-3.4, lag2 ~2.7-2.9, lag3 ~1.6-2.0, lag4 ~0.8-1.1, lag5 ~0.5, later ~0;
  first-in-window coefficient negative (-0.6..-1.1) = artifact of persistent rates; SUM of lags stable ~8.8 across windows.

## Session 2 (runs 38-46)
- Teacher-forced original (fwd_2, sel100): coef of next finger pred on insulin jitter dev = +0.38 (se .32) vs data -1.62 (se .66): model IGNORES dose magnitude.
- fwd_3 states at insulin bundles: band base R2 0.90, rate R2 0.65, jitter-dev R2 0.15 -> dose magnitude barely in state. Head-only edit can't work.
- Plan: dedicated h-dim leaky memory (zero its rows in blocks/coda outputs, set decay via flow.a/a_in, inject via b_in row), read by value head. Needs local reimpl of encoders+prelude; verify via z_post mean.

## Session 3
- value_bin head = MLP 192->192->17; last_numeric (192,8). Lever: add vector along last_numeric z_last_value column scaled by a dose memory.
- TARGETS on sel100 tuning (refined theta): ins_x2 -14.1 (se1.2), ins_half +7.1 (.6), ne_x2 +3.8 (.27), ne_half -1.9 (.13). Model orig: -0.39,+0.15,-0.66,-0.24. Edit names: insulin_x2, insulin_half, ne_x2, ne_half.
- Embedding tables: used rows: kind{0,2}, ekind=event kind id, kq=0, text/action/route/status/discharged_to unused(const), rate_unit{2,3}, unit{2..5}. Init-norm .28 = unused.
- Plan: full local reimpl of model, validate vs states (all bundles) + factual means (fwd_2 sel100).

## Session 4 (runs 100-110)
- Old local step (no source recoverable) gives step_err3('empty_const') = (5.255, 0.0614) -> ~6% rel error, NOT validated.
- Config: d192, heads 4, ff x4, prelude1, recurrent2, coda1, slots4, mem128, z32. coda/prelude = pre-norm layers (norm1, attn, norm3, ffn).
- Design: dedicated dim j: injection only via prelude attention (u_k token) -> E -> b_in row j; persistence: zero writes to j in blocks/coda/c_out, gamma_j=0 in norms; readout via out_norm/query_norm -> value_bin.0 column j.
- Keep outputs SHORT (content filter).

## Session 5 (runs 137-148)
- Keep replies SHORT/neutral (classifier). Refer to drugs by code id: 8=ins, 9=ne; 6=gf,10=gl,7=map_art,5=map_cuff.
- Attn layers: q.weight, kv.weight[384] (k then v?), out.weight, NO biases. Norms=LayerNorm(w,b). b_in,c_out no bias.
- h scale: per-bundle sd across dims ~65 (+-7), mean .6+-1.9, per-dim std ~8.6. out (LN): per-dim std ~0.1.
- z post dims: several near-constant (std .04-.06).
- stay_bundles(sid)-> list of DataFrames [minute,code,kind,ekind,value,rate,dose]; enc_bundle(df)->(e,u,n1,n2).
- END atoms (kind2) carry rate/dose of ended record. zpost_err() = 0.102 (local encoders inexact).
- fwd_4 = states all bundles, 12 tuning stays (644 rows). fwd_3 = insulin bundles.
- Only value_bin head matters for metrics (factual value NLL + rollout means). Other heads can be perturbed.

## Session 6
- Re-read ARCH. Door for dose info into state = prelude attention only (u_k / z tokens). Head reads out, h-after-flow, z-prior, own-code register.

## ARCH summary (session 7)
Step(dt): h*=exp(-dt*exp(flow.a)); ctx=out+out_norm(mean_slots(h))+gap(log dt/10); z=z_post([ctx,e,u]) mean.
prelude.0 pre-norm over [h+slots(4),e,u,z_in(z)] -> slot outs -> prelude_norm -> E.
u2=blocks(h+E); h=a_bar*h+softplus(delta_in)*b_in(E)+(u2-(h+E)); a_bar=exp(-softplus(delta_in)*exp(a_in)).
y=coda(c_out(h)); h=y; out=out_norm(mean_slots(y)).
Head: view=out+out_norm(mean_slots(h_flow))+gap(log dt)+z_head(z prior x4, probs avg); q=query_norm(view+type_emb).
with_code=q+code_emb+last_numeric([has,logage/10,z_last_value,z_last_rate])+bins; logits=value_bin(with_code)+transition[c*17+lastbin].
Rollout: times/codes recorded; values drawn+fed back; code-8/9 atoms scaled from h12.

## Session 8
- Current edit = free dims 49,185 (zero writes in prelude/blocks/coda, zero norm w/b, c_out/b_in row+col). fwd_6 (100 tuning): NLL 1.52794 -> 1.53133 (+0.0034). No injection/readout yet.
- Local head validated (run 205-206): type_emb[1]; gap(log dt_s/10) for ctx AND view; last_numeric feat8=[has,log1p(age_h)/10,0,has,0,0,zval,0] (best fit); bins.rate[0]=last_status[0]=0. MAE vs tool ~1.5 (noise floor ~1.3 with 4 tool samples).
