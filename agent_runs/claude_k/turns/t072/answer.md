The model's prediction errors confirm the hypothesis.

| Drug | Error coefficient, lag 1 | Lag 2 | Later lags |
|---|---|---|---|
| Insulin → glucose (per unit/h of random dose part) | −1.47 ± 0.29 | −1.43 ± 0.30 | near zero |
| Norepinephrine → MAP | +2.5 ± 0.9 | +2.5 ± 0.9 | similar, noisy |

So the model misses roughly −1.5 for insulin at each of lags 1–2. Cross effects are near zero.

For norepinephrine the coefficients are noisier and spread over later lags. That spread is expected, because a norepinephrine dose shock carries over into later rates (titration builds on the previous rate). Undoing that carryover puts the missing direct effect mostly at lag 1, about +2.5 to +3.2 per mcg/kg/min. Both estimates match my earlier comparison of the causal estimates with the model's own rollout sensitivity.

The plan, then: a gated value-head correction driven by a linear probe of q. The probe target is the dose minus the dose the readings alone imply, weighted by the missing kernel. Built that way, it averages to zero in factual data and responds fully to a changed dose.

The probe needs more data than 99 stays, so I am choosing 300 more training stays with heavy use of both drugs, for state passes. This run picks them and updates my notebook.

```python
sel_set = set(int(s_) for s_ in sel_states) | set(int(s_) for s_ in sel1)
ids_arr = np.array(ids)
cand_idx = np.where((nins >= 14) & (nne >= 14))[0]
cand_ids = [int(ids_arr[i_]) for i_ in cand_idx if int(ids_arr[i_]) not in sel_set]
print('candidates with both drugs >=14h:', len(cand_ids))
rng2 = np.random.default_rng(7); new300 = sorted(rng2.choice(cand_ids, 300, replace=False).tolist())
json.dump(new300, open('/work/new300.json', 'w'))
print(json.dumps(new300[:100]))
with open('/work/notes.md', 'a') as fh:
    fh.write("""
## Reconstruction findings (runs 42-67)
- ALL MLPs are GELU (not ReLU). LayerNorm eps 1e-5. type_emb row 1 for value atoms. gap input log(dt_s)/10. Time flow exp(-dt_s*exp(flow.a)) applies for heads view.
- z_post order [ctx,e,u]; unit table rows: glucose 4, MAP 2; my posterior recon corr ~0.85 (not exact).
- value head recon: last_numeric layout fitted (A2, 8x4 on [1,has,age,zlast]); held-out MAE 1.3 vs tool predicted means.
- fwd3: factual 3 stays (mistake). fwd4: factual original on 99 states stays (value NLL 1.520).
- State after insulin bundle encodes dose residual (dose - band-expected) with CV R2 ~0.42 (out/slotmean/h).
- Model factual error regressed on dose shocks: insulin lag1 -1.47(.29), lag2 -1.43(.30), lag3-4 ~0 ; norepi lags1-4 2.5,2.5,1.4,2.5 (+-0.9) (shock persists in rates).
- PLAN: repurpose never-active value-head hidden units (GELU) as gated linear units reading a probe of q for dose-minus-readings-implied-dose; calibrate gains with dose_edit.
""")
```