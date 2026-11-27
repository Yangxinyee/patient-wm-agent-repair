Two things from ARCHITECTURE.md change the design.

**1. The coda is part of the state update.** "The new h is y", where y = coda(c_out(h)). So a memory dimension must be passed through unchanged by `c_out` (identity on that row) and by the coda (no writes to it). It also appears in `out`, and therefore in `ctx` and the head's view. I also need to cut its reads by `z_prior`/`z_post`, so the memory does not leak into z.

**2. The value head's attribute input is fixed.** It is `with_code = q + embed.code[code] + last_numeric(...) + embed.bins.value[last bin] + embed.bins.rate[...] + last_status`. `embed.code` and `embed.bins` are shared with the encoder, so I cannot freely change them. But at a repurposed dimension j, their contribution is a fixed function of (code, last bin). That gives a natural code gate: pick the memory dimension j where `embed.code[glucose_finger, j]` and `embed.code[map_arterial, j]` differ strongly. A GELU unit reading `with_code_j` is then in its linear range for glucose and switched off for MAP. Its constant offsets can be cancelled in `embed_transition` rows, which are indexed by exactly (code, last bin).

Only the value head matters for both the factual NLL and the rollout readouts. Side effects on the action and time heads therefore do not count.

**Next step:** check which dimensions give clean gating, and how much "common-mode" contamination LayerNorm adds. Specifically:
- the per-dimension code-embedding gaps (glucose vs MAP);
- the spread of bin embeddings at each dimension;
- how stable `mean/std` of h is across states, since that sets the contamination;
- how the replica's step and evaluator are called, so I can extract E for fitting the write decoder.

**What would confirm or refute the gating idea:**
- Confirmed: some dimensions where the code gap is well above the bin-embedding spread.
- Refuted: no such dimension. Then I would need a separate gate dimension.

```python
with open('/work/notes.md','a') as fh_:
    fh_.write("""
## Restart 4 (runs 106-111)
- ARCH facts: abar/bundle ~0.30-0.45 ALL dims (state rewritten each bundle via E->b_in); sp~1; flow decay/hr 0.0025..0.9999 (dt in s); NEW h = y = coda(c_out(h_n)); out=out_norm(mean y).
- No dead h dims (min std 2.27 dim1; hbar across-dim std ~65, cv .11) -> memory must repurpose dims; LN perturbation small if |m|<~30.
- with_code = q + embed.code[code] + last_numeric + bins.value[lastbin] + bins.rate[lastrate] + last_status; value logits = value_bin MLP + embed_transition[code*17+lastbin]. Only value head matters for eval.
- PLAN: memory dim j per drug (abar=1 via a_in<<0, flow.a -> decay/hr, b_in row j = linear decoder of dev from E, zero all other reads/writes of j in prelude/blocks/c_out/coda/z_post/z_prior/gap/z_head/last_numeric/type_emb, value MLP: zero W0[:,j] for old units, repurpose 1 unit k: GELU(a*with_code_j+b0) gated by code_emb[c,j] separation; W2[:,k]=tilt; compensate offsets in embed_transition).
- tf_eval(P, build_ds(tr, stays, 0), False) -> (nll[n], probs[n,17], ?) 0.1s/10 stays.
""")
C_ = load_original('embed.code.weight').float(); BV = load_original('embed.bins.value.weight').float(); BR = load_original('embed.bins.rate.weight').float()
print("codes:", VOCAB['codes'] if isinstance(VOCAB['codes'],list) else list(VOCAB['codes'])[:13])
print("cid:", cid)
gf, gl, ma, mc = cid['glucose_finger'], cid['glucose_lab'], cid['map_arterial'], cid['map_cuff']
dG = C_[[gf,gl]].min(0).values - C_[[ma,mc]].max(0).values     # glucose higher than both MAP
dM = C_[[ma,mc]].min(0).values - C_[[gf,gl]].max(0).values
bvr = BV[1:].max(0).values - BV[1:].min(0).values
print("BR rows std over dims (row0 vs others):", BR.std(1)[:4])
sc = torch.maximum(dG, dM)/bvr
o = torch.argsort(-sc)[:12]
print("best gating dims (dim, glucose-minus-MAP gap, MAP-minus-glucose gap, bins.value range, ratio):")
for j in o: print(int(j), round(float(dG[j]),3), round(float(dM[j]),3), round(float(bvr[j]),3), round(float(sc[j]),2))
hb = HS.float().mean(1); r_m = hb.mean(1)/hb.std(1); print("mu/sd of hbar across states: mean %.4f std %.4f"%(r_m.mean(), r_m.std()))
for fn in [tf_eval, tf_states_ds, build_ds, step_b, rollout_gen]:
    print(fn.__name__, "names:", fn.__code__.co_names[:40], "| locals:", fn.__code__.co_varnames[:30])
```