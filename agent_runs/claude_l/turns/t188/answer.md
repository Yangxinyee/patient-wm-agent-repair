The architecture document gives the exact update rule:

`h ← a_bar·h + softplus(delta_in)·b_in(E) + (block increments)`, then `h ← coda(c_out(h))`.

Before that there is a per-dimension time decay with timescales from 10 minutes up to 2 years. My earlier measurement fits this: `a_bar` ≈ 0.27–0.47 per step, and the block and coda pass rewrites the state, so a dose perturbation fades within about an hour. Two points matter for the edit:

1. `z_post` is effectively unused. The heads draw z from the prior, and the prelude barely attends to the z token. So `z_post` and `z_in` can be repurposed. They are also the only place where the context (which predicts the policy's dose) meets the dose token, which makes them the natural place to compute the dose residual.
2. A longer dose memory would need a channel in `h` that the blocks and coda leave intact, with a decay I can set. Whether that exists depends on the shapes and form of `a_in`, `delta_in`, `c_out` and the coda.

Next I'll print my exact replica code (`wm2.py`, `wm2_rollout.py`) and the shapes of the core, z and value-head tensors.
- If `a_in`/`delta_in` are fixed per-dimension parameters and the coda is a residual MLP, a persistent channel is plausible.
- If the coda normalises and mixes every dimension, I'll fall back to calibrating a one-step gain.

```python
print(open('/work/wm2.py').read())
print("=====")
print(open('/work/wm2_rollout.py').read())
print("=====")
for n in NAMES:
    if n.startswith('core.') and ('blocks' not in n) or n.startswith('z_') or n.startswith('out_norm') or n.startswith('core.coda') or n.startswith('attr_heads.value_bin') or n.startswith('query_norm'):
        print(n, SHAPES[n])
```