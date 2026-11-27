My local re-implementation only covers the encoders, the context and the z posterior. It does not run the prelude, the recurrent blocks or the coda, so I cannot yet test locally how a dose signal would spread through the core.

The prelude has no token-type embedding: its tokens are h plus core.slots, e_k, u_k and z_in(z_k). The rest is standard pre-norm attention and FFN with residuals.

Next I will write a local core step and check it against the model's real states saved in fwd_2. For that I need the head count and the remaining tensor shapes (z_in, gap, attribute heads). This run prints the config and the non-core tensors.

- **Confirms the plan:** the config gives n_heads and memory details, so a core step can be built.
- **Refutes it:** key details are missing. Then I would fall back to measuring the channel empirically with "states" passes.

```python
import json
print(json.dumps(VOCAB.get('config',{}), indent=None)[:1500])
other=[n for n in NAMES if not (n.startswith('core') or n.startswith('encode_action') or n.startswith('encode.') or 'pending' in n or 'filter' in n)]
print(len(other))
print('; '.join(f"{n}{tuple(SHAPES[n])}" for n in other))
```