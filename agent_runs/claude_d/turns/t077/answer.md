The vocab confirms the head layout: 4 heads, ff_mult 4, one prelude layer, two recurrent layers, one coda layer, z_dim 32, and decay timescales from 10 minutes to 2 years. It does not say which rows of the categorical tables an atom uses. So an exact numpy re-implementation of the action encoder and z_post, which a centred z_post feature would need, carries real risk and many runs. The last attempt reached only about 2.9 error.

Before committing to that, I'll measure whether centring is actually needed and what each lag is missing. A factual pass on the same 100 training stays would let me regress the reading residuals (reading minus the model's predicted mean) on the policy residual at each lag. Since that residual is independent of everything else, the regression gives directly how much the model under-responds at each lag in teacher-forced mode. It also gives the reading-level noise, which tells me how much factual budget an uncentred edit would use.

First I'll record the findings so far in the notebook.

```python
with open('/work/notes.md','a') as f_: f_.write("""
## Restart 2 (runs 58-71)
- NOTE: content filter blocked 2 outputs; keep prints short/technical.
- Fwd1 per-hour (sel1, ins_x2): model shift h13..24 = -0.61..-2.52 (mean -1.73) vs true -3.2..-17.9 (mean -14.0). Hour-13 (first, no feedback) model/dose12 = -0.45 (median -0.31) vs true lag1 -1.99 => dose->state pathway itself 4-6x too weak, accumulation even weaker.
- beta_ins indexed lag0..12 with lag0=0, lag1=-1.99 (dose at h affects reading at h+1).
- Fwd3 = states original, 100 random train stays (selS in /work/selS.json; mix ins-only 858, ne-only 942, both 503 bundles). 5192 rows.
  has_insulin includes END-only bundles (rate 0, ~9%). Frac-hour .17 bundles = 2-atom non-drug (100 of them).
  Ridge R2 at 1237 true insulin bundles: dose surprise (eps~rate-1.03*base) from z 0.00, out 0.12, h 0.31; rate z .29/out .62/h .71; base z .48/out .88/h .88.
  => z posterior does NOT encode dose surprise; state encodes base (glucose-implied dose) strongly, actual dose weakly.
- Arch facts: z_post.0 (192,576) reads [ctx,e_k,u_k]; z_in (192,32) only in state update; z_head uses prior samples. config: 4 heads, ff 4x, 1 prelude, 2 recurrent, 1 coda.
- Idea under consideration: centred edit needed (uncentred dose amplification biases factual preds by ~(c-c')E[dose|H]).
""")
print('ok')
```