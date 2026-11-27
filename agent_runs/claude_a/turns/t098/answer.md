The architecture document arrived intact, and it settles how dose information can reach the glucose and MAP predictions.

**What it shows.** The only route from an infusion record into the state is:

1. the action set encoder produces u;
2. u enters the prelude, either as its own attention token or through the z posterior;
3. the prelude output E is written into h through `b_in` and the recurrent blocks;
4. h passes through `c_out` and the coda.

The memory M stores only measurement and visit bundle vectors e, so it cannot hold doses. The value head reads the state only through LayerNormed summaries. The "last value" register it uses is per code, so it carries the previous reading of the same code and nothing about doses.

**What I conclude.**
- Any dose memory that lasts beyond one hour has to live in dimensions of h.
- The write path has to pass through the prelude attention.
- The readout should be a code-gated logit tilt in the value head. That tilt can be offset exactly through the per-code rows of the transition table.

**What I do next.** I will list all tensor names and shapes so I can see the attention, FFN and MLP layouts that a repurposed channel would have to pass through. I will also save the architecture summary to the notebook, since the conversation may restart.

**How this outcome decides the plan.**
- *Favourable:* the shapes show modest hidden sizes and per-dimension gates (`a_in`, `delta_in`, `flow.a`) that make a dedicated h dimension possible.
- *Unfavourable:* the gates are scalars shared across dimensions. Then I would need a different persistence mechanism.

```python
print(' '.join(f'{n}{tuple(SHAPES[n])}'.replace(' ','') for n in NAMES))
with open('/work/notes.md','a') as fh_:
    fh_.write("""
## Restart 3 (run 92-94): architecture recap (ARCHITECTURE.md)
- atom vec = code+kind+ekind+unit cats + bins.value/dose/rate + numeric@[z_value,has_value,z_dose,has_dose,z_rate,has_rate]+bias
- DeepSets: norm(mlp2(sum mlp1(atom)/sqrt(n) + size(log1p n))). meas/visit -> e_k (also stored in memory M); insulin/NE -> u_k (0 if none). M holds ONLY e => no dose info in M.
- step: h*=exp(-dt exp(flow.a)); z: ctx=out+out_norm(mean_slots h)+gap; post([ctx,e,u]) mean used; prelude over [h+slots,e,u,z_in] -> prelude_norm -> E;
  x=blocks(h+E); h=abar*h+softplus(delta_in)*b_in(E)+(x-(h+E)), abar=exp(-softplus(delta_in)exp(a_in)); y=coda(c_out(h)); out=out_norm(mean_slots y)
- head: view=out+out_norm(mean_slots(h flowed))+gap+z_head(z~prior); q=query_norm(view+type_emb); with_code=q+code_emb+last_numeric+bins.value[lastbin]+bins.rate[lastrate]+last_status
  value logits = value_bin MLP(with_code) + embed_transition.value_bin[code*17+lastbin]
- Only route dose->state: u (or z_post) -> PRELUDE ATTENTION -> E -> h. Plan idea: dedicated memory dims in h (write via prelude/b_in, persist via flow.a/abar, zero other writers), readout = code-gated GELU unit in value MLP giving logit tilt; constant offset compensated in embed_transition rows of that code.
- TF compensation idea: memory should hold (rate - expected rate | glucose tier) so factual mean unchanged; e.g. write -f(G) at reading bundle, +r at action bundle.
""")
```