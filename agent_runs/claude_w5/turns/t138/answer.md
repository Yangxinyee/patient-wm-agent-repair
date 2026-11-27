My kernel holds what I need for a local head check:
- the 100 training stays of fwd_2/fwd_3 as bundles (`bundles_B`);
- their tool states (`X_`, `SIDS`);
- weights in float64 (`W64`);
- the exact core step `cstep`, whose wiring is visible in `tf_check`.

The head has three unknowns: the `type_emb` row (15 rows), the 8-input `last_numeric` layout, and the `last_status` row. I will first save the recent findings to the notebook, since the conversation restarts. Then I will look at the head tables for clues: the norms of the 8 `last_numeric` columns, which `type_emb` rows look trained, and the `last_status` rows. A column or row that was never trained should have a small, untouched norm (about 0.27, like the unused embedding rows), which narrows the layout before I fit against fwd_3's predicted means.

```python
with open('/work/notes.md','a') as fh:
    fh.write("""
## Runs 122-128 (after restart 2)
- Run 122: initial state = zeros (h, out) with first dt = 1 s reproduces first state row (err 9e-8). Local core exact.
- ARCHITECTURE: a_bar fixed per dim (params only); new h = y = coda(c_out(h_new)); readout = head predicted MEAN (sum p(b)*centre), edited - factual, 20 paths.
- Shapes: value head = Linear192-GELU-Linear17; z_post 576->192->64 (only mean 32 used in core); FFNs 768; b_in, c_out 192x192 no bias; last_numeric (192,8) (8 inputs, layout unknown); type_emb 15 rows; last_status 2 rows.
- fwd_3 = factual original on the 100 train stays of fwd_2 (selB): NLL/value 1.4364 (gf 1.538, gl 1.304, map_art 1.331, cuff 1.597). Columns: stay_id,hour,event,observed,predicted_mean,nll.
- Kernel vars: bundles_B (100 stays), X_ (fwd_2 states), SIDS, W64 (float64 weights), cstep, enc2, tf_check (loop template).
- PLAN: verify head locally vs fwd_3 predicted means; then build edit = residualized dose -> glucose/MAP belief shift at drug bundles,
  calibrated so rollout effect matches g-est (ins x2 ~ -14.3, x1/2 +7.15, ne x2 +4.11, x1/2 -2.05 on tuning sel A).
""")
Wl = W64['last_numeric.weight']; print('last_numeric col norms', np.round(np.linalg.norm(np.asarray(Wl),axis=0),3))
Te = np.asarray(W64['type_emb.weight']); print('type_emb row norms', np.round(np.linalg.norm(Te,axis=1),3))
Ls = np.asarray(W64['last_status.weight']); print('last_status row norms', np.round(np.linalg.norm(Ls,axis=1),3))
zpw = np.asarray(W64['z_prior.2.bias']); print('z_prior logsd bias range', np.round(zpw[32:].min(),2), np.round(zpw[32:].max(),2))
```