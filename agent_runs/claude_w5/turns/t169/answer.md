The local head is exact up to sampling noise once the prior's second half is read as a log-std. With 4 draws, my mean NLL is 1.434–1.439 across seeds, bracketing fwd_3's 1.4364. My RMSE against fwd_3 (1.83) equals the RMSE between two of my own 4-draw runs (1.80). So I now have a complete, verified local model: core plus value head.

The natural next step is to reproduce the dose_edit rollout locally, so I can calibrate an edit without spending forward passes. For that I need the exact rollout rules: what is drawn, how readings are sampled from the bins, and which rows count. These are in the last lines of ARCHITECTURE.md. I'll record the head result in the notebook first.

```python
with open('/work/notes.md','a') as fh:
    fh.write("""
## Runs 137-154 (after restart 3)
- Prelude attention at drug bundles: slots attend ~100% to u token (heads 1,2 =1.0; heads 0,3 ~0.96). At reading bundles: head0->e 0.99, heads1,3->u(empty) 0.95, head2->slots.
- |y| ~1792 (huge), abar=exp(-sp(delta_in)*exp(a_in)) ~0.36, sp(delta_in)~1, flow decay/h median 0.98.
- ins x2 at one bundle: u cos 0.97, attn_out change ~122 (|attn_out| 530), E rel change 7.5%, but y rel 0.5%, out rel 0.3%.
- HEAD VERIFIED: type_emb index 1 for readings, last_status row 0, last_numeric cols: 0=has_last, 3=log1p(age_s)/10, 6=z_last_value (others 0).
  view = out_prev + out_norm(mean(y_prev*exp(-dt*exp(flow.a)))) + gap(log(dt)/10) + z_head(z); z = mu + exp(s)*eps from z_prior(view0) (2nd half = LOG-STD).
  4-draw local NLL 1.434-1.439 vs fwd3 1.4364; rmse vs fwd3 = rmse local-vs-local (~1.8). Kernel: H (5673 rows), V0, fixed, TRr, cen, probs_from_v0().
""")
for j in range(79, 92):
    if j < len(txt): print(j, txt[j][:230])
```