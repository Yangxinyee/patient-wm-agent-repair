Results:
- The insulin and NE rate and dose bins **saturate**: insulin's top bin is ≥5 U/h; NE's top bins are ≥1.0, its cap in the records. Only the numeric z-scores keep growing linearly, up to a clip at ±10.
- `c_out` is a general matrix, not near the identity, so there is no free linear memory in h.
- All 8 `last_numeric` inputs are in use.
- The helper functions show I had re-implemented the embeddings, the set encoders, ctx and the z posterior (GELU activations), but not the output heads.

**Design decision.** The edit will add a code-gated linear readout in the value head:
- a decoder, fitted by regression, estimates from the head input the departure of recent doses from the policy-expected dose;
- one repurposed hidden unit per drug turns that estimate into a tilt of the bin logits, for glucose codes (insulin) or MAP codes (NE);
- outcome-level compensation, if needed, goes into the transition table, which is keyed by the last reading's bin.

To fit this I need my own re-implementation of the head that reproduces the tool's teacher-forced predicted means. That requires factual output on the same stays whose states I already have (the 100 training stays of fwd_3). This run saves notes and prints those stay ids and my earlier variant settings.

```python
notes_add = """
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
"""
open('/work/notes.md','a').write(notes_add)
print(sorted(set(S['stay_id'].tolist())) == sorted(selT))
print(json.dumps([int(s_) for s_ in selT]))
print({k_: globals().get(k_) for k_ in ['gapmode','act','side','unit_row','half','variant','gscale','eps','bad','sid0']})
```