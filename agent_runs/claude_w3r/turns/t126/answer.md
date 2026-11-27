This answers where the dose signal is lost. In the prelude, attention head 2 attends entirely to the u token and head 1 entirely to the e token. Heads 0 and 3 spread evenly over the slots, and nothing attends to z.

| Stage | Top singular value | What it means |
|---|---|---|
| u → slot tokens (head 2) | 114 | strong, in one main direction |
| u → E (after prelude_norm, slot std ≈ 65) | 1.5 | u still reaches E well |
| E → `out` | ≤ 0.053 | the bottleneck |

So the dose does reach E. It is lost in the step from E to `out`, because the slow state y has very large values, so out_norm divides any change by about 58.

A workable channel therefore needs:
- the dose to be linearly readable in E;
- an amplified E→h path in that one direction, for example a rank-1 addition to core.b_in.

Before building it I will save these findings to my notebook, since a restart may come soon.

```python
notes_add = """
## Session 5 (restart 5) - local CORE STEP implemented (runs 95-117)
- Local core step works (validated vs fwd_2 states, starting from true prev state and model posterior z):
  time flow h*=exp(-dt_s*exp(flow.a)); prelude tokens [h+slots(4), e, u, z_in(z)] (7 tokens always, no mask);
  pre-norm layer (norm1->attn, norm3->ffn), E=prelude_norm(slot tokens); blocks x=h+E: per layer x+=selfattn(norm1 x); x+=cross(norm2 x, Mem+AE, Mem+AE)
  where Mem = e vectors of ALL bundles incl current (last 128), AE = core.age(log1p(age_seconds)) ADDED TO MEMORY before kv (both k and v);
  x+=ffn(norm3 x); hn=abar*h+sp*b_in(E)+(x-(h+E)); y=c_out(hn); y+=attn(coda.norm1 y); y+=ffn(coda.norm3 y); out=out_norm(mean_slots(y)).
  4 heads x 48, no attn biases. END atoms -> OBS encoder (e); start atoms -> encode_action (u). Unit rows: mg/dL=3, mmHg=4 (better), insulin 4->?, norepi 5.
  Accuracy: out median abs err 0.022, h rel err 0.045 (action bundles better than measurement bundles).
- h is HUGE (slot norm ~800, a few dims 100-200), slots nearly identical. LN scales ~58-65 everywhere.
- PRELUDE ATTENTION: head0, head3 -> slots uniformly; head1 -> e token 100%; head2 -> u token 100%; nobody attends z.
- Jacobians at insulin bundle: dXslot/du top sv 114 (then 4.5); dE/du top sv 1.48; dout/du 0.049 (rank-1); dout/dE <= 0.053; dout/dz ~0.
  => u reaches E fine (via head2) but E->out is attenuated (LN of huge y). z posterior has ~no effect on state.
- PLAN: find E-direction carrying insulin rate linearly (low normal variance), amplify E->h via rank-1 edit of core.b_in (or c_out),
  read at gf value head via dead units (value_bin.0 has 17 dead units) with tilt vector; compensate band mean -b1*m(g).
  Persistence of injected h-component (abar~.3-.44/step) may supply lag-2/3 terms; glucose feedback supplies the rest (avoid double counting).
"""
with open('/work/notes.md','a') as f_: f_.write(notes_add)
print('ok')
```