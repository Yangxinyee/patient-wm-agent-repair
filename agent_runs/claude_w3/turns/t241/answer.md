The file's second half gives the whole state path, so I'll restate it neutrally.
- Each step runs the prelude to get E, then the recurrent blocks.
- The h update is `a_bar*h + softplus(delta_in)*b_in(E) + block residual`.
- The coda then gives the new h, and `out` comes from it.
- The value head reads `out`, a normalised copy of the time-decayed h, the gap, z from the prior, and a register of the same code's last value.
- In the edited rollout, the reading times are as recorded. Values are drawn by the model and fed back, and only the infusion atoms are scaled.

So any effect of the code-8/code-9 rate on later readings has to be carried in h, entering through the prelude. Next I'll save this summary to the notebook so it survives restarts, and list the `core.*` tensor shapes. I need to know whether the decay terms are per dimension or per slot before designing a memory channel. If `flow.a`, `a_in` and `delta_in` turn out to be per-dimension vectors of length 192, then one shared dimension can serve as a leaky memory.

```python
s='''
## ARCH summary (session 7)
Step(dt): h*=exp(-dt*exp(flow.a)); ctx=out+out_norm(mean_slots(h))+gap(log dt/10); z=z_post([ctx,e,u]) mean.
prelude.0 pre-norm over [h+slots(4),e,u,z_in(z)] -> slot outs -> prelude_norm -> E.
u2=blocks(h+E); h=a_bar*h+softplus(delta_in)*b_in(E)+(u2-(h+E)); a_bar=exp(-softplus(delta_in)*exp(a_in)).
y=coda(c_out(h)); h=y; out=out_norm(mean_slots(y)).
Head: view=out+out_norm(mean_slots(h_flow))+gap(log dt)+z_head(z prior x4, probs avg); q=query_norm(view+type_emb).
with_code=q+code_emb+last_numeric([has,logage/10,z_last_value,z_last_rate])+bins; logits=value_bin(with_code)+transition[c*17+lastbin].
Rollout: times/codes recorded; values drawn+fed back; code-8/9 atoms scaled from h12.
'''
open('/work/notes.md','a').write(s)
for n in NAMES:
    if n.startswith('core.') and ('blocks' not in n) and ('prelude.0' not in n) and ('coda.0' not in n):
        print(n, tuple(SHAPES[n]))
print([n for n in NAMES if 'blocks.0.0' in n])
```