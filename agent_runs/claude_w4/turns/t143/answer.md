Heads 1–3 differ in how they treat non-drug bundles:
- **Head 1** attends 100% to the u_k token at every bundle, drug or not (minimum 1.0 at measurement bundles, 0.998 at drug bundles). Its output is therefore a clean, state-independent function of the drug atoms. A rank-1 change to its value weights can add any linear readout of LN1(u_k), zero at the no-action vector, into all four slots without leaking from other tokens.
- **Head 2** looks at u_k at drug bundles but at the slots elsewhere.
- **Head 3** is mixed at measurement bundles.

What remains open is the factual fit. A clean injection of κ·rate would bias factual predictions unless I subtract the expected rate given the state. That needs a gated, state-dependent term, which the prelude FFN could supply.

I'll record these findings in the notebook before going further.

```python
notes_add = """
## Runs 104-122 (restart 3): v1 FAILED; prelude attention mapped
- fwd_5 = v1 (z_post injection) dose_edit tuning sel100: ins x2 -0.349, x1/2 +0.209, NE x2 +0.018 -> identical to original. v1 useless.
- WHY: fwd_4 probe (z_in.weight=0) vs original states: out rel change 0.2%, h <1% -> z token barely used by state update.
- Prelude CPU replica (pre-norm, 4 heads x48, kv rows 0:192=k,192:384=v, exact GELU, LN eps1e-5). Slot->token attention:
  drug bundles: head0 slots .95/e .05; heads1,2,3 -> u_k token .998/.997/.992; z token 0 everywhere.
  measurement bundles (approx e_k): head0 slots .91/e .09; HEAD1 -> u token 1.00 (min 1.0); head2 slots .99; head3 slots .6/u .28/e .11.
  => HEAD 1 = clean always-on u_k reader: output = Wo[:,48:96] V1 LN1(u_k), state-independent. No-action u_const = LN(mlp2(size.bias)) (norm 11.6).
- PLAN v2: carrier via head-1 value path (rank-1 add to kv rows 240:288) and/or prelude FFN units (gate+state E) to inject kappa*(rate - E[rate|state]).
  Factual fit needs subtraction of E[rate|state]; linear readout weak (R2 .56) vs band ceiling ~.8 -> consider step units on latest-G readout.
"""
with open('/work/notes.md','a') as f: f.write(notes_add)
print('ok')
```